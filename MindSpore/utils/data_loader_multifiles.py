import logging
import glob
import random
import numpy as np
import h5py
import mindspore.context as context
from utils.img_utils import reshape_fields, reshape_precip
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size, context

def get_data_loader(params, files_pattern, distributed, train):

    dataset = GetDataset(params, files_pattern, train, distributed)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if params.mae_pretrain:
        define_column_names = ['img', 'ids_keep', 'mask', 'ids_restore']
    else:
        define_column_names = ['prev', 'future']
    sampler = None
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        dataloader = ds.GeneratorDataset(dataset, 
                                    column_names=define_column_names, column_types=None, schema=None, num_samples=len(dataset), 
                                    num_parallel_workers=params.num_data_workers, shuffle=train, sampler=sampler, num_shards=get_group_size(), shard_id=get_rank(), 
                                    python_multiprocessing=True, max_rowsize=100)
    else:
        dataloader = ds.GeneratorDataset(dataset, 
                                    column_names=define_column_names, column_types=None, schema=None, num_samples=len(dataset), 
                                    num_parallel_workers=params.num_data_workers, shuffle=train, sampler=sampler, num_shards=1, shard_id=0, 
                                    python_multiprocessing=True, max_rowsize=100)

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

class GetDataset:
    def __init__(self, params, location, train, distributed, ):
        self.params = params
        self.debug = params['debug']
        self.distributed = distributed
        self.location = location
        self.train = train
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.roll = params.roll
        self._get_files_stats()
        self.two_step_training = params.two_step_training
        self.orography = params.orography
        self.precip = params.precip_mode
        self.add_noise = params.add_noise if train else False
        self.mae_pretrain = params.mae_pretrain
        self.mask_ratio = params.mask_ratio
        

        if self.precip:
            path = params.precip+'/train' if train else params.precip+'/test'
            self.precip_paths = glob.glob(path + "/*.h5")
            logging.info(f"start precip: precip data num: {len(self.precip_paths)}; data: {self.precip_paths}")
            self.precip_paths.sort()

        try:
            self.normalize = params.normalize
        except:
            self.normalize = True #by default turn on normalization if not specified in config
            
        if self.orography:
            self.orography_path = params.orography_path

        _patchify_out_shape_x = int((self.img_shape_x-params.patch_size[0] + 2*0)/params.patch_size[0] + 1)
        _patchify_out_shape_y = int((self.img_shape_y-params.patch_size[1] + 2*0)/params.patch_size[1] + 1)
        self.patchify_blocks_num = _patchify_out_shape_x * _patchify_out_shape_y
        self.len_keep = int(self.patchify_blocks_num * (1 - self.mask_ratio))

        logging.info("Patchify Image Shape: {} x {}; Blocks Num: {}".format(
            _patchify_out_shape_x, _patchify_out_shape_y, self.patchify_blocks_num))
        logging.info(f"self.precip: {self.precip}; self.two_step_training: {self.two_step_training}; self.mae_pretrain: {self.mae_pretrain}")

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        logging.info(f"load data from {self.location}")
        logging.info(f"_get_files_stats: {self.files_paths}")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            #original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] -1#just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        if self.debug:
            logging.info(f"debug mode: train mode? {self.train}")
            logging.info(f"debug mode: all files: {self.files_paths}")
            if self.distributed:
                logging.info(f"debug mode: n_samples_per_year from {self.n_samples_per_year} to 36")
                self.n_samples_per_year = min(4*4*8, self.n_samples_per_year)
            else:
                logging.info(f"debug mode: n_samples_per_year from {self.n_samples_per_year} to 3")
                # self.n_samples_per_year = 500
                self.n_samples_per_year = 36

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        self.precip_files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
        logging.info("Delta t: {} hours".format(6*self.dt))
        logging.info("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))


    def clean_files(self):
        try:
            self.files = [None for _ in range(self.n_years)]
            self.precip_files = [None for _ in range(self.n_years)]
            return 1
        except:
            return 0

    def _open_file(self, year_idx):
        if self.debug:
            logging.info(f"debug mode: open file from {self.files_paths[year_idx]}")
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']  
        if self.orography:
            _orog_file = h5py.File(self.orography_path, 'r')
            self.orography_field = _orog_file['orog']
        if self.params.precip_mode:
            self.precip_files[year_idx] = h5py.File(self.precip_paths[year_idx], 'r')['tp']
        
  
    def __len__(self):
        return self.n_samples_total


    def __getitem__(self, global_idx):
       
        year_idx = int(global_idx/self.n_samples_per_year) #which year we are on
        local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

        y_roll = np.random.randint(0, 1440) if self.train else 0#roll image in y direction

        #open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        if not self.precip:
            #if we are not at least self.dt*n_history timesteps into the prediction
            if local_idx < self.dt*self.n_history:
                local_idx += self.dt*self.n_history

            #if we are on the last image in a year predict identity, else predict next timestep
            step = 0 if local_idx >= self.n_samples_per_year-self.dt else self.dt
        else:
            inp_local_idx = local_idx
            tar_local_idx = local_idx
            #if we are on the last image in a year predict identity, else predict next timestep
            step = 0 if tar_local_idx >= self.n_samples_per_year-self.dt else self.dt
            # first year has 2 missing samples in precip (they are first two time points)
            if year_idx == 0:
                lim = 1458
                local_idx = local_idx%lim 
                inp_local_idx = local_idx + 2
                tar_local_idx = local_idx
                step = 0 if tar_local_idx >= lim-self.dt else self.dt

        #if two_step_training flag is true then ensure that local_idx is not the last or last but one sample in a year
        if self.two_step_training:
            if local_idx >= self.n_samples_per_year - 2*self.dt:
                local_idx = self.n_samples_per_year - 3*self.dt

        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0

        if self.orography:
            orog = self.orography_field[0:720] 
        else:
            orog = None

        if self.train and (self.crop_size_x or self.crop_size_y):
            rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
            rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
        else: 
            rnd_x = 0
            rnd_y = 0
            
        if self.precip:
            return reshape_fields(self.files[year_idx][inp_local_idx, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train), \
                    reshape_precip(self.precip_files[year_idx][tar_local_idx+step], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)
        else:
            if self.two_step_training:
                return reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                        reshape_fields(self.files[year_idx][local_idx + step:local_idx + step + 2, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)
            elif self.mae_pretrain:
                noise = np.random.rand(self.patchify_blocks_num)
                ids_shuffle = np.argsort(noise, axis=0).astype(np.int32)
                ids_restore = np.argsort(ids_shuffle, axis=0).astype(np.int32)
                ids_keep = ids_shuffle[:self.len_keep]

                mask = np.ones(self.patchify_blocks_num, dtype=np.float16)
                mask[:self.len_keep] = 0
                mask = np.take_along_axis(mask, ids_restore, axis=0)
                
                img = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise)
                return img, ids_keep, mask, ids_restore
            else:
                return reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                        reshape_fields(self.files[year_idx][local_idx + step, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)
