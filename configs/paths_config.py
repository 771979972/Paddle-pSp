# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("Paddle_pSp")

dataset_paths = {
	'celeba_train': '',
	'celeba_test': 'datasets/CelebA_test/',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': 'datasets/FFHQ/',
}
# 'celeba_test': 'CelebA/img_align_celeba/',
model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pdparams',
	'ir_se50': 'pretrained_models/model_ir_se50.pdparams',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pdparams',
	'alexnet': 'pretrained_models/alexnet.pdparams',
	'lin_alex0.1': 'pretrained_models/lin_alex.pdparams',
	'mtcnn_pnet': 'models/mtcnn/mtcnn_paddle/src/weights/pnet.npy',
	'mtcnn_rnet': 'models/mtcnn/mtcnn_paddle/src/weights/rnet.npy',
	'mtcnn_onet': 'models/mtcnn/mtcnn_paddle/src/weights/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
