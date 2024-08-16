"""Define the dataset which are used in the scripts
"""
from abc import abstractmethod, ABC
import glob
from dataclasses import dataclass
import os
import pathlib
import random
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import trimesh
from said.util.audio import load_audio
from said.util.blendshape import load_blendshape_coeffs, load_blendshape_deltas
from said.util.mesh import create_mesh, get_submesh, load_mesh
from said.util.parser import parse_list


def check_exist(file_path):
    if not os.path.exists(file_path):
        print('{} not exist'.format(file_path))
        return False

    return True


def get_frames(image_dir, if_return_npy=False):
    image_list = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    image_list.sort()

    return image_list


def get_data_paths(group_names, data_dirs, augment_nums=None, lest_video_frames=30, mode='train'):
    data_dic_name_list = []
    person_id = 0
    sentence_id = 0

    for data_dir, group_name, augment_num in zip(data_dirs, group_names, augment_nums):
        print('== start loading data from {} and group {} =='.format(data_dir, group_name))

        person_id += 1

        group_dir = os.path.join(data_dir, group_name)
        if mode == 'train':
            file_path = os.path.join(group_dir, 'train.txt')
        else:
            file_path = os.path.join(group_dir, 'val.txt')

        print('== start loading data from {} =='.format(file_path))

        with open(file_path,'r', encoding='utf-8') as f:
            data = f.readlines()
            data = data * augment_num
            print('== {} data num is {} =='.format(group_name, len(data)))

        # crop_frames_path = os.path.join(os.path.dirname(file_path), '3dmm_crop_frames')
        cpem_path = os.path.join(group_dir, 'cpem')
        images_path = os.path.join(cpem_path, 'train_images')
        masks_path = os.path.join(cpem_path, 'train_masks')
        origin_images_path = os.path.join(cpem_path, 'origin_images')
        roi_path = os.path.join(cpem_path, 'roi')
        # for A2 model train
        audio_dir = os.path.join(group_dir, 'audio')
        bs_dir = os.path.join(cpem_path, 'bs')

        for line in tqdm(data):
            prefix = line.strip()
            train_image_frame_dir = os.path.join(images_path, prefix)
            mask_frame_dir = os.path.join(masks_path, prefix)
            ori_image_dir = os.path.join(origin_images_path, prefix)
            audio_path = os.path.join(audio_dir, prefix + '.wav')
            if not os.path.exists(audio_path):
                print('== {} not exist =='.format(audio_path))
                continue

            bs_npy_path = os.path.join(bs_dir, prefix + '.npy')
            if not os.path.exists(bs_npy_path):
                print('== {} not exist =='.format(bs_npy_path))
                continue
            
            sentence_id += 1

            roi_pickle_path = os.path.join(roi_path, prefix + '.pkl')
            # roi_list = get_roi_list(roi_pickle_path)
            roi_list = []
            # check exist
            if check_exist(train_image_frame_dir) and check_exist(mask_frame_dir) and check_exist(ori_image_dir):
                train_image_list, mask_image_list, ori_image_list = get_frames(train_image_frame_dir), get_frames(mask_frame_dir), get_frames(ori_image_dir)
                frame_num = min([len(train_image_list), len(mask_image_list), len(ori_image_list)])
                roi_list = [p.replace('train_images', 'roi').replace('.png', '.npy') for p in train_image_list]

                # check frames
                if frame_num < lest_video_frames:
                    print('=====> {} frames less than {}'.format(prefix, lest_video_frames))
                    continue

                # if len(roi_list) != frame_num:
                #     print('=====> {} roi list len less than {}'.format(len(roi_list), frame_num))
                #     continue

                key_str = group_name + '|' + prefix
                # data_dic_name_list.append(key_str)
                # roi_npy_path = os.path.join(roi_path, prefix + '.npy')
                # roi_npy_path = os.path.join(roi_path, prefix + '.pkl')

                data = BlendVOCADataPath(
                            person_id=person_id,
                            sentence_id=sentence_id,
                            audio=audio_path,
                            blendshape_coeffs=bs_npy_path,
                        )
                data_dic_name_list.append(data)
                
            else:
                continue

    random.shuffle(data_dic_name_list)
    print('finish loading')

    return data_dic_name_list


@dataclass
class DataItem:
    """Dataclass for the data item"""
    waveform: Optional[torch.FloatTensor]  # (audio_seq_len,)
    blendshape_coeffs: Optional[
        torch.FloatTensor
    ]  # (blendshape_seq_len, num_blendshapes)
    cond: bool = True  # If false, uncondition for classifier-free guidance
    blendshape_delta: Optional[torch.FloatTensor] = None  # (num_blendshapes, |V|, 3)
    person_id: Optional[str] = None
    sentence_id: Optional[int] = None


@dataclass
class DataBatch:
    """Dataclass for the data batch"""

    waveform: List[np.ndarray]  # List[(audio_seq_len,)] with length batch_size
    blendshape_coeffs: Optional[
        torch.FloatTensor
    ]  # (batch_size, blendshape_seq_len, num_blendshapes)
    cond: torch.BoolTensor  # (batch_size,)
    blendshape_delta: Optional[
        torch.FloatTensor
    ] = None  # (batch_size, num_blendshapes, |V|, 3)
    person_ids: Optional[List[str]] = None
    sentence_ids: Optional[List[int]] = None


@dataclass
class ExpressionBases:
    """Dataclass for the expression bases (including neutral)"""

    neutral: trimesh.Trimesh  # Neutral mesh object
    blendshapes: Dict[str, trimesh.Trimesh]  # <blendshape name>: blendshape mesh object


@dataclass
class BlendVOCADataPath:
    """Dataclass for the BlendVOCA data path"""

    person_id: str
    sentence_id: int
    audio: Optional[str]
    blendshape_coeffs: Optional[str]


class BlendVOCADataset(ABC, Dataset):
    """Abstract class of BlendVOCA dataset"""

    fps = 60

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> DataItem:
        """Return the item of the given index

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        DataItem
            DataItem object
        """
        pass

    
    def get_blend_npy(self, blend_fpath):
        blend_npy = np.load(blend_fpath)
        # select mouth blend npy
        blend_npy = blend_npy[:, -27:]

        return torch.from_numpy(blend_npy).float()


    @staticmethod
    def collate_fn(examples: List[DataItem]) -> DataBatch:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[DataItem]
            List of the outputs of __getitem__

        Returns
        -------
        DataBatch
            DataBatch object
        """
        waveforms = [np.array(item.waveform) for item in examples]
        blendshape_coeffss = None
        if len(examples) > 0 and examples[0].blendshape_coeffs is not None:
            blendshape_coeffss = torch.stack(
                [item.blendshape_coeffs for item in examples]
            )
        conds = torch.BoolTensor([item.cond for item in examples])
        blendshape_deltas = None
        if len(examples) > 0 and examples[0].blendshape_delta is not None:
            blendshape_deltas = torch.stack(
                [item.blendshape_delta for item in examples]
            )

        person_ids = None
        if len(examples) > 0 and examples[0].person_id is not None:
            person_ids = [item.person_id for item in examples]

        sentence_ids = None
        if len(examples) > 0 and examples[0].sentence_id is not None:
            sentence_ids = [item.sentence_id for item in examples]

        return DataBatch(
            waveform=waveforms,
            blendshape_coeffs=blendshape_coeffss,
            cond=conds,
            blendshape_delta=blendshape_deltas,
            person_ids=person_ids,
            sentence_ids=sentence_ids,
        )

    @staticmethod
    def preprocess_blendshapes(
        templates_dir: str,
        blendshape_deltas_path: str,
        blendshape_indices: Optional[List[int]] = None,
        person_ids: Optional[List[str]] = None,
        blendshape_classes: Optional[List[str]] = None,
    ) -> Dict[str, ExpressionBases]:
        """Preprocess the blendshapes

        Parameters
        ----------
        templates_dir : str
            Directory path of the templates
        blendshape_deltas_path : str
            Path of the blendshape deltas file
        blendshape_indices : Optional[List[int]], optional
            List of the blendshape indices, by default None
        person_ids : Optional[List[str]], optional
            List of the person ids, by default None
        blendshape_classes : Optional[List[str]], optional
            List of the blendshape classes, by default None

        Returns
        -------
        Dict[str, ExpressionBases]
            {
                <Person id>: expression bases
            }
        """
        if blendshape_indices is None:
            blendshape_indices_path = (
                pathlib.Path(__file__).parent.parent.parent
                / "data"
                / "FLAME_head_idx.txt"
            )
            blendshape_indices = parse_list(blendshape_indices_path, int)

        if person_ids is None:
            person_ids = (
                BlendVOCADataset.person_ids_train
                + BlendVOCADataset.person_ids_val
                + BlendVOCADataset.person_ids_test
            )

        if blendshape_classes is None:
            blendshape_classes = BlendVOCADataset.default_blendshape_classes

        blendshape_deltas = load_blendshape_deltas(blendshape_deltas_path)

        expressions = {}
        for pid in tqdm(person_ids):
            template_mesh_path = os.path.join(templates_dir, f"{pid}.ply")
            template_mesh_ori = load_mesh(template_mesh_path)
            submesh_out = get_submesh(
                template_mesh_ori.vertices, template_mesh_ori.faces, blendshape_indices
            )

            vertices = submesh_out.vertices
            faces = submesh_out.faces

            neutral_mesh = create_mesh(vertices, faces)

            bl_deltas = blendshape_deltas[pid]

            blendshapes_dict = {}
            for bl_name in blendshape_classes:
                bl_vertices = vertices + bl_deltas[bl_name]
                blendshapes_dict[bl_name] = create_mesh(bl_vertices, faces)

            expressions[pid] = ExpressionBases(
                neutral=neutral_mesh, blendshapes=blendshapes_dict
            )

        return expressions


class BlendVOCATrainDataset(BlendVOCADataset):
    """Train dataset for VOCA-ARKit"""
    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        blendshape_deltas_path: Optional[str],
        landmarks_path: Optional[str],
        sampling_rate: int,
        window_size_min: int = 25,
        uncond_prob: float = 0.1,
        zero_prob: float = 0,
        hflip: bool = True,
        delay: bool = True,
        delay_thres: int = 1,
        # classes: List[str] = BlendVOCADataset.default_blendshape_classes,
        # classes_mirror_pair: List[
        #     Tuple[str, str]
        # ] = BlendVOCADataset.default_blendshape_classes_mirror_pair,
        preload: bool = True,
    ) -> None:
        """Constructor of the class

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        blendshape_deltas_path : Optional[str]
            Path of the blendshape deltas
        landmarks_path: Optional[str]
            Path of the landmarks data
        sampling_rate : int
            Sampling rate of the audio
        window_size_min : int, optional
            Minimum window size of the blendshape coefficients, by default 120
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        zero_prob : float, optional
            Zero-out probability of waveform and blendshape coefficients, by default 0
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        delay : bool, optional
            Whether do the delaying, by default True
        delay_thres: int, optional
            Maximum amount of delaying, by default 1
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        preload: bool, optional
            Load the data in the constructor, by default True
        """
        self.sampling_rate = sampling_rate
        self.window_size_min = window_size_min
        self.uncond_prob = uncond_prob
        self.zero_prob = zero_prob

        self.hflip = hflip
        self.delay = delay
        self.delay_thres = delay_thres
        
        data_dirs = ['/data/xxx']
        group_names = ['xxx']
        # data augmentation
        augment_nums = [1]

        self.data_paths = get_data_paths(group_names, data_dirs, augment_nums, lest_video_frames=30, mode='train')
        self.max_bs_lens = 1000

        self.preload = preload
        self.data_preload = []
        if self.preload:
            print('Preloading data...')
            for data in tqdm(self.data_paths):
                waveform = load_audio(data.audio, self.sampling_rate)
                blendshape_coeffs = self.get_blend_npy(data.blendshape_coeffs)
                self.data_preload.append((waveform, blendshape_coeffs))


    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]

        if self.preload:
            data_pre = self.data_preload[index]
            waveform = data_pre[0]
            blendshape_coeffs = data_pre[1]
        else:
            waveform = load_audio(data.audio, self.sampling_rate)
            blendshape_coeffs = self.get_blend_npy(data.blendshape_coeffs)

        # cut too long waveformif blendshape_coeffs.shape[0] > self.max_bs_lens
        if blendshape_coeffs.shape[0] > self.max_bs_lens:
            blendshape_coeffs = blendshape_coeffs[:self.max_bs_lens, :]
            waveform = waveform[:self.max_bs_lens * 4 * 160]

        # Random uncondition for classifier-free guidance
        cond = random.uniform(0, 1) > self.uncond_prob

        # Random zero-out
        if random.uniform(0, 1) < self.zero_prob:
            waveform = torch.zeros_like(waveform)
            blendshape_coeffs = torch.zeros_like(blendshape_coeffs)

        return DataItem(
            waveform=waveform,
            blendshape_coeffs=blendshape_coeffs,
            cond=cond,
            blendshape_delta=None,
        )

    def collate_fn(self, examples: List[DataItem]) -> DataBatch:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[DataItem]
            List of the outputs of __getitem__

        Returns
        -------
        DataBatch
            DataBatch object
        """
        conds = torch.BoolTensor([item.cond for item in examples])

        blendshape_deltas = None
        if len(examples) > 0 and examples[0].blendshape_delta is not None:
            blendshape_deltas = torch.stack(
                [item.blendshape_delta for item in examples]
            )

        person_ids = None
        if len(examples) > 0 and examples[0].person_id is not None:
            person_ids = [item.person_id for item in examples]

        sentence_ids = None
        if len(examples) > 0 and examples[0].sentence_id is not None:
            sentence_ids = [item.sentence_id for item in examples]

        waveforms = [item.waveform for item in examples]
        blendshape_coeffss = [item.blendshape_coeffs for item in examples]

        bc_min_len = min([coeffs.shape[0] for coeffs in blendshape_coeffss])
        window_size = random.randrange(self.window_size_min, bc_min_len + 1)

        # waveform_window_len = (self.sampling_rate * window_size) // self.fps
        waveform_window_len = window_size * 4 * self.sampling_rate // 100
        batch_size = len(waveforms)

        half_window_size = window_size

        waveforms_windows = []
        coeffs_windows = []
        # Random-select the window
        for idx in range(batch_size):
            waveform = waveforms[idx]
            blendshape_coeffs = blendshape_coeffss[idx]

            blendshape_len = blendshape_coeffs.shape[0]
            num_blendshape = blendshape_coeffs.shape[1]

            # print('waveform: ', waveform.shape)
            # normalize the waveformif blendshape_len * 4 > waveform.shape[0] // 160:
            if blendshape_len * 4 > waveform.shape[0] // 160:
                waveform = F.pad(waveform, (0, blendshape_len * 4 * 160 - waveform.shape[0]))
            else:
                waveform = waveform[:blendshape_len * 4 * 160]

            # 
            bdx = random.randint(
                0, max(0, blendshape_len - window_size - 1)
            )

            bdx_update = bdx
            coeffs_window = blendshape_coeffs[bdx_update : bdx_update + window_size, :]

            if self.delay:
                if random.uniform(0, 1) < 0.5:
                    waveform_window = waveform[bdx_update * 4 * self.sampling_rate // 100 : (bdx_update + window_size) * 4 * self.sampling_rate // 100]
                    waveform_window = F.pad(waveform_window, (2 * self.delay_thres * 160, 0), mode='constant', value=0)
                else:
                    waveform_window = waveform[bdx_update * 4 * self.sampling_rate // 100 : (bdx_update + window_size) * 4 * self.sampling_rate // 100]
                    waveform_window = F.pad(waveform_window, (0, 2 * self.delay_thres * 160), mode='constant', value=0)
            else:
                waveform_window = waveform[bdx_update * 4 * self.sampling_rate // 100 : (bdx_update + window_size) * 4 * self.sampling_rate // 100]
                waveform_window = F.pad(waveform_window, (160, 160), mode='constant', value=0)

            waveforms_windows.append(waveform_window)
            coeffs_windows.append(coeffs_window)

        coeffs_final = torch.stack(coeffs_windows)
        waveforms_final = [np.array(waveform) for waveform in waveforms_windows]

        return DataBatch(
            waveform=waveforms_final,
            blendshape_coeffs=coeffs_final,
            cond=conds,
            blendshape_delta=blendshape_deltas,
            person_ids=person_ids,
            sentence_ids=sentence_ids,
        )


class BlendVOCAValDataset(BlendVOCADataset):
    """Validation dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        blendshape_deltas_path: Optional[str],
        landmarks_path: Optional[str],
        sampling_rate: int,
        uncond_prob: float = 0.1,
        zero_prob: float = 0,
        hflip: bool = True,
        # classes: List[str] = BlendVOCADataset.default_blendshape_classes,
        # classes_mirror_pair: List[
        #     Tuple[str, str]
        # ] = BlendVOCADataset.default_blendshape_classes_mirror_pair,
        preload: bool = True,
    ) -> None:
        """Constructor of the class

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        blendshape_deltas_path : Optional[str]
            Path of the blendshape deltas
        landmarks_path: Optional[str]
            Path of the landmarks data
        sampling_rate : int
            Sampling rate of the audio
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        zero_prob : float, optional
            Zero-out probability of waveform and blendshape coefficients, by default 0
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        preload: bool, optional
            Load the data in the constructor, by default True
        """
        self.sampling_rate = sampling_rate
        self.window_size_min = 25
        self.uncond_prob = uncond_prob
        self.zero_prob = zero_prob

        self.hflip = hflip
        # self.delay = delay
        # self.delay_thres = delay_thres
        
        data_dirs = ['/data/xxx']
        group_names = ['xxx']
        # data augmentation
        augment_nums = [1]
        
        self.max_bs_lens = 1000

        self.data_paths = get_data_paths(group_names, data_dirs, augment_nums, lest_video_frames=30, mode='val')

        self.preload = preload
        self.data_preload = []
        if self.preload:
            print('Preloading data...')
            for data in tqdm(self.data_paths):
                waveform = load_audio(data.audio, self.sampling_rate)
                blendshape_coeffs = self.get_blend_npy(data.blendshape_coeffs)
                self.data_preload.append((waveform, blendshape_coeffs))


    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]

        if self.preload:
            data_pre = self.data_preload[index]
            waveform = data_pre[0]
            blendshape_coeffs = data_pre[1]
        else:
            waveform = load_audio(data.audio, self.sampling_rate)
            blendshape_coeffs = self.get_blend_npy(data.blendshape_coeffs)

        # cut too long waveformif blendshape_coeffs.shape[0] > self.max_bs_lens
        if blendshape_coeffs.shape[0] > self.max_bs_lens:
            blendshape_coeffs = blendshape_coeffs[:self.max_bs_lens, :]
            waveform = waveform[:self.max_bs_lens * 4 * 160]

        # Random uncondition for classifier-free guidance
        cond = random.uniform(0, 1) > self.uncond_prob

        # Random zero-out
        if random.uniform(0, 1) < self.zero_prob:
            waveform = torch.zeros_like(waveform)
            blendshape_coeffs = torch.zeros_like(blendshape_coeffs)

        return DataItem(
            waveform=waveform,
            blendshape_coeffs=blendshape_coeffs,
            cond=cond,
            blendshape_delta=None,
        )

    def collate_fn(self, examples: List[DataItem]) -> DataBatch:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[DataItem]
            List of the outputs of __getitem__

        Returns
        -------
        DataBatch
            DataBatch object
        """
        conds = torch.BoolTensor([item.cond for item in examples])

        blendshape_deltas = None
        if len(examples) > 0 and examples[0].blendshape_delta is not None:
            blendshape_deltas = torch.stack(
                [item.blendshape_delta for item in examples]
            )

        person_ids = None
        if len(examples) > 0 and examples[0].person_id is not None:
            person_ids = [item.person_id for item in examples]

        sentence_ids = None
        if len(examples) > 0 and examples[0].sentence_id is not None:
            sentence_ids = [item.sentence_id for item in examples]

        waveforms = [item.waveform for item in examples]
        blendshape_coeffss = [item.blendshape_coeffs for item in examples]

        bc_min_len = min([coeffs.shape[0] for coeffs in blendshape_coeffss])
        window_size = bc_min_len

        # waveform_window_len = (self.sampling_rate * window_size) // self.fps
        waveform_window_len = window_size * 4 * self.sampling_rate // 100
        batch_size = len(waveforms)

        half_window_size = window_size

        waveforms_windows = []
        coeffs_windows = []
        # Random-select the window
        for idx in range(batch_size):
            waveform = waveforms[idx]
            blendshape_coeffs = blendshape_coeffss[idx]

            blendshape_len = blendshape_coeffs.shape[0]
            num_blendshape = blendshape_coeffs.shape[1]

            # print('waveform: ', waveform.shape)
            # normalize the waveformif blendshape_len * 4 > waveform.shape[0] // 160:
            if blendshape_len * 4 > waveform.shape[0] // 160:
                waveform = F.pad(waveform, (0, blendshape_len * 4 * 160 - waveform.shape[0]))
            else:
                waveform = waveform[:blendshape_len * 4 * 160]

            # 
            bdx = random.randint(
                0, max(0, blendshape_len - window_size - 1)
            )

            bdx_update = bdx
            coeffs_window = blendshape_coeffs[bdx_update : bdx_update + window_size, :]

            waveform_window = waveform[bdx_update * 4 * self.sampling_rate // 100 : (bdx_update + window_size) * 4 * self.sampling_rate // 100]
            waveform_window = F.pad(waveform_window, (160, 160), mode='constant', value=0)

            waveforms_windows.append(waveform_window)
            coeffs_windows.append(coeffs_window)

        coeffs_final = torch.stack(coeffs_windows)
        waveforms_final = [np.array(waveform) for waveform in waveforms_windows]

        return DataBatch(
            waveform=waveforms_final,
            blendshape_coeffs=coeffs_final,
            cond=conds,
            blendshape_delta=blendshape_deltas,
            person_ids=person_ids,
            sentence_ids=sentence_ids,
        )


class BlendVOCATestDataset(BlendVOCADataset):
    """Test dataset for BlendVOCA"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: Optional[str],
        blendshape_deltas_path: Optional[str],
        sampling_rate: int,
        preload: bool = True,
    ) -> None:
        """Constructor of the class

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        blendshape_deltas_path : Optional[str]
            Path of the blendshape deltas
        sampling_rate : int
            Sampling rate of the audio
        preload: bool, optional
            Load the data in the constructor, by default True
        """
        self.sampling_rate = sampling_rate

        self.data_paths = self.get_data_paths(
            audio_dir, blendshape_coeffs_dir, self.person_ids_test
        )

        self.blendshape_deltas = (
            load_blendshape_deltas(blendshape_deltas_path)
            if blendshape_deltas_path
            else None
        )

        self.preload = preload
        self.data_preload = []
        self.blendshape_deltas_preload = {}
        if self.preload:
            for data in self.data_paths:
                waveform = load_audio(data.audio, self.sampling_rate)
                blendshape_coeffs = (
                    load_blendshape_coeffs(data.blendshape_coeffs)
                    if data.blendshape_coeffs
                    else None
                )
                self.data_preload.append((waveform, blendshape_coeffs))

                if data.person_id not in self.blendshape_deltas_preload:
                    blendshape_delta = (
                        torch.FloatTensor(
                            np.stack(
                                list(self.blendshape_deltas[data.person_id].values()),
                                axis=0,
                            )
                        )
                        if self.blendshape_deltas
                        else None
                    )

                    self.blendshape_deltas_preload[data.person_id] = blendshape_delta

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]

        if self.preload:
            data_pre = self.data_preload[index]
            waveform = data_pre[0]
            blendshape_coeffs = data_pre[1]
            blendshape_delta = self.blendshape_deltas_preload[data.person_id]
        else:
            waveform = load_audio(data.audio, self.sampling_rate)
            blendshape_coeffs = (
                load_blendshape_coeffs(data.blendshape_coeffs)
                if data.blendshape_coeffs
                else None
            )
            blendshape_delta = (
                torch.FloatTensor(
                    np.stack(
                        list(self.blendshape_deltas[data.person_id].values()), axis=0
                    )
                )
                if self.blendshape_deltas
                else None
            )

        waveform_window = waveform
        if blendshape_coeffs is not None:
            blendshape_len = blendshape_coeffs.shape[0]
            waveform_window_len = (self.sampling_rate * blendshape_len) // self.fps

            # Adjust the waveform window
            waveform_tmp = waveform[:waveform_window_len]

            waveform_window = torch.zeros(waveform_window_len)
            waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]

        return DataItem(
            waveform=waveform_window,
            blendshape_coeffs=blendshape_coeffs,
            blendshape_delta=blendshape_delta,
        )


class BlendVOCAEvalDataset(BlendVOCADataset):
    """Evaluation dataset for BlendVOCA"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        blendshape_deltas_path: Optional[str],
        sampling_rate: int,
        # classes: List[str] = BlendVOCADataset.default_blendshape_classes,
        preload: bool = True,
        repeat_regex: str = "(-.+)?",
    ):
        """Constructor of the class

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        blendshape_deltas_path : Optional[str]
            Path of the blendshape deltas
        sampling_rate : int
            Sampling rate of the audio
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        preload: bool, optional
            Load the data in the constructor, by default True
        repeat_regex: str, optional
            Regex for checking the repeated files, by default "(-.+)?"
        """
        self.sampling_rate = sampling_rate
        self.classes = classes

        self.data_paths = self.get_data_paths(
            audio_dir,
            blendshape_coeffs_dir,
            self.person_ids_test,
            repeat_regex,
        )

        self.preload = preload
        self.data_preload = []

        if self.preload:
            for data in self.data_paths:
                waveform = load_audio(data.audio, self.sampling_rate)
                blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)
                self.data_preload.append((waveform, blendshape_coeffs))


    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]

        if self.preload:
            data_pre = self.data_preload[index]
            waveform = data_pre[0]
            blendshape_coeffs = data_pre[1]
            blendshape_delta = self.blendshape_deltas_preload[data.person_id]
        else:
            waveform = load_audio(data.audio, self.sampling_rate)
            blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)
            blendshape_delta = (
                torch.FloatTensor(
                    np.stack(
                        list(self.blendshape_deltas[data.person_id].values()), axis=0
                    )
                )
                if self.blendshape_deltas
                else None
            )

        blendshape_len = blendshape_coeffs.shape[0]
        waveform_window_len = (self.sampling_rate * blendshape_len) // self.fps

        # Adjust the waveform window
        waveform_tmp = waveform[:waveform_window_len]

        waveform_window = torch.zeros(waveform_window_len)
        waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]

        return DataItem(
            waveform=waveform_window,
            blendshape_coeffs=blendshape_coeffs,
            blendshape_delta=blendshape_delta,
            person_id=data.person_id,
            sentence_id=data.sentence_id,
        )


class BlendVOCAPseudoGTOptDataset:
    """Dataset for generating pseudo-GT blendshape coefficients"""

    def __init__(
        self,
        neutrals_dir: str,
        blendshapes_dir: str,
        mesh_seqs_dir: str,
        blendshapes_names: List[str],
    ) -> None:
        """Constructor of the BlendVOCAPseudoGTOptDataset

        Parameters
        ----------
        neutrals_dir : str
            Directory which contains the neutral meshes
        blendshapes_dir : str
            Directory which contains the blendshape meshes
        mesh_seqs_dir : str
            Directory which contains the mesh sequences
        blendshapes_names : List[str]
            List of the blendshape names
        """
        self.neutrals_dir = neutrals_dir
        self.blendshapes_dir_dir = blendshapes_dir
        self.mesh_seqs_dir_dir_dir = mesh_seqs_dir
        self.blendshapes_names = blendshapes_names

    def get_blendshapes(self, person_id: str) -> ExpressionBases:
        """Return the dictionary of the blendshape meshes

        Parameters
        ----------
        person_id : str
            Person id that wants to get the blendshapes

        Returns
        -------
        ExpressionBases
            Expression bases object
        """
        neutral_path = os.path.join(self.neutrals_dir, f"{person_id}.obj")
        blendshapes_dir = os.path.join(self.blendshapes_dir_dir, person_id)

        neutral_mesh = load_mesh(neutral_path)

        blendshapes_dict = {}
        for bl_name in self.blendshapes_names:
            bl_path = os.path.join(blendshapes_dir, f"{bl_name}.obj")
            bl_mesh = load_mesh(bl_path)
            blendshapes_dict[bl_name] = bl_mesh

        return ExpressionBases(neutral=neutral_mesh, blendshapes=blendshapes_dict)

    def get_mesh_seq(self, person_id: str, seq_id: int) -> List[trimesh.base.Trimesh]:
        """Return the mesh sequence

        Parameters
        ----------
        person_id : str
            Person id
        seq_id : int
            Sequence id

        Returns
        -------
        List[trimesh.base.Trimesh]
            List of the meshes
        """
        mesh_seq_dir = os.path.join(
            self.mesh_seqs_dir_dir_dir, person_id, f"sentence{seq_id:02}"
        )

        if not os.path.isdir(mesh_seq_dir):
            return []

        files_obj = glob.glob(os.path.join(mesh_seq_dir, "**/*.obj"), recursive=True)
        files_ply = glob.glob(os.path.join(mesh_seq_dir, "**/*.ply"), recursive=True)

        mesh_seq_paths = sorted(files_obj + files_ply)

        mesh_seq_list = []
        for seq_path in mesh_seq_paths:
            mesh = load_mesh(seq_path)
            mesh_seq_list.append(mesh)

        return mesh_seq_list


class BlendVOCAVAEDataset(BlendVOCADataset):
    """Abstract class of BlendVOCA dataset for VAE"""

    def __init__(
        self,
        blendshape_coeffs_dir: str,
        window_size: int = 120,
        zero_prob: float = 0,
        hflip: bool = True,
        dataset_type: str = "train",
        # classes: List[str] = BlendVOCADataset.default_blendshape_classes,
        # classes_mirror_pair: List[
        #     Tuple[str, str]
        # ] = BlendVOCADataset.default_blendshape_classes_mirror_pair,
    ) -> None:
        """Constructor of the class

        Parameters
        ----------
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        window_size : int, optional
            Window size of the blendshape coefficients, by default 120
        zero_prob : float, optional
            Zero-out probability of waveform and blendshape coefficients, by default 0
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        dataset_type: str, optional
            Type of the dataset, whether "train", "val", and "test", by default "train"
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        """
        self.window_size = window_size
        self.zero_prob = zero_prob

        self.hflip = hflip
        self.classes = classes
        self.classes_mirror_pair = classes_mirror_pair

        self.mirror_indices = []
        self.mirror_indices_flip = []
        for pair in self.classes_mirror_pair:
            index_l = self.classes.index(pair[0])
            index_r = self.classes.index(pair[1])
            self.mirror_indices.extend([index_l, index_r])
            self.mirror_indices_flip.extend([index_r, index_l])

        person_ids = None
        if dataset_type == "train":
            person_ids = self.person_ids_train
        elif dataset_type == "val":
            person_ids = self.person_ids_val
        else:
            person_ids = self.person_ids_test

        self.data_paths = self.get_data_paths(blendshape_coeffs_dir, person_ids)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]
        blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)

        num_blendshape = blendshape_coeffs.shape[1]
        blendshape_len = blendshape_coeffs.shape[0]

        half_window_size = self.window_size // 2

        # Random-select the window
        bdx = random.randint(
            -half_window_size, max(0, blendshape_len - half_window_size - 1)
        )
        bdx_update = bdx + half_window_size
        coeffs_window = F.pad(
            blendshape_coeffs.unsqueeze(0),
            (0, 0, half_window_size, self.window_size),
            "replicate",
        ).squeeze(0)[bdx_update : bdx_update + self.window_size, :]

        """
        bdx = random.randint(0, max(0, blendshape_len - self.window_size))
        coeffs_tmp = blendshape_coeffs[bdx : bdx + self.window_size, :]
        coeffs_window = torch.zeros((self.window_size, num_blendshape))
        coeffs_window[: coeffs_tmp.shape[0], :] = coeffs_tmp[:]
        """

        # Augmentation - hflip
        if self.hflip and random.uniform(0, 1) < 0.5:
            coeffs_window[:, self.mirror_indices] = coeffs_window[
                :, self.mirror_indices_flip
            ]

        # Random zero-out
        if random.uniform(0, 1) < self.zero_prob:
            coeffs_window = torch.zeros_like(coeffs_window)

        return DataItem(
            waveform=None,
            blendshape_coeffs=coeffs_window,
        )

    # def get_data_paths(
    #     self,
    #     blendshape_coeffs_dir: str,
    #     person_ids: List[str],
    # ) -> List[BlendVOCADataPath]:
    #     """Return the list of the data paths

    #     Parameters
    #     ----------
    #     blendshape_coeffs_dir : str
    #         Directory of the blendshape coefficients
    #     person_ids : List[str]
    #         List of the person ids

    #     Returns
    #     -------
    #     List[BlendVOCADataPath]
    #         List of the BlendVOCADataPath objects
    #     """
    #     data_paths = []

    #     for pid in person_ids:
    #         coeffs_id_dir = os.path.join(blendshape_coeffs_dir, pid)
    #         if coeffs_id_dir is None or not os.path.exists(coeffs_id_dir):
    #             continue

    #         for sid in self.sentence_ids:
    #             filename_base = f"sentence{sid:02}"
    #             coeffs_pattern = re.compile(f"^{filename_base}(-.+)?\.csv$")
    #             filename_list = [
    #                 s for s in os.listdir(coeffs_id_dir) if coeffs_pattern.match(s)
    #             ]
    #             for filename in filename_list:
    #                 coeffs_path = os.path.join(coeffs_id_dir, filename)
    #                 if os.path.exists(coeffs_path):
    #                     data = BlendVOCADataPath(
    #                         person_id=pid,
    #                         sentence_id=sid,
    #                         audio=None,
    #                         blendshape_coeffs=coeffs_path,
    #                     )
    #                     data_paths.append(data)

    #     return data_paths

    @staticmethod
    def collate_fn(examples: List[DataItem]) -> DataBatch:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[DataItem]
            List of the outputs of __getitem__

        Returns
        -------
        DataBatch
            DataBatch object
        """
        blendshape_coeffss = None
        if len(examples) > 0 and examples[0].blendshape_coeffs is not None:
            blendshape_coeffss = torch.stack(
                [item.blendshape_coeffs for item in examples]
            )
        conds = torch.BoolTensor([item.cond for item in examples])

        return DataBatch(
            waveform=[],
            blendshape_coeffs=blendshape_coeffss,
            cond=conds,
        )
