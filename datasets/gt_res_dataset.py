#!/usr/bin/python
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
import os
from paddle.io import Dataset
from PIL import Image


class GTResDataset(Dataset):

	def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None):
		self.pairs = []
		for f in os.listdir(root_path):
			image_path = os.path.join(root_path, f)
			gt_path = os.path.join(gt_dir, f)
			if f.endswith(".jpg") or f.endswith(".png"):
				# self.pairs.append([image_path, gt_path.replace('.png', '.jpg'), None])
				self.pairs.append([image_path, gt_path, None])
		self.transform = transform
		self.transform_train = transform_train

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		from_path, to_path, _ = self.pairs[index]
		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')

		if self.transform:
			to_im = self.transform(to_im)
			from_im = self.transform(from_im)

		return from_im, to_im
