from __future__ import absolute_import

# import sys
# sys.path.append('./')

import os
from PIL import Image, ImageFile
import numpy as np
import random
import json
import lmdb
import sys
import six

import torch
from torch.utils import data
from torch.utils.data import sampler
from torchvision import transforms

from lib.utils.labelmaps import get_vocabulary, labels2strs
from lib.utils import to_numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


from config import get_args
global_args = get_args(sys.argv[1:])

if global_args.run_on_remote:
  import moxing as mox


class CustomDataset(data.Dataset):
  def __init__(self, root, gt_file_path, embed_path, voc_type, max_len, num_samples, transform=None):
    super(CustomDataset, self).__init__()
    self.root = root
    self.embed_path = embed_path
    self.voc_type = voc_type
    self.transform = transform
    self.max_len = max_len
    # self.gt_file = json.load(open(gt_file_path, "r"))
    # self.nSamples = len(self.gt_file)
    # self.nSamples = min(self.nSamples, num_samples)

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)
    self.lowercase = (voc_type == 'LOWERCASE')

    if os.path.basename(gt_file_path).split(".")[-1] == "json":
      self.images_path, self.transcriptions, self.embeds_path = self.load_gt_json(gt_file_path)
    elif os.path.basename(gt_file_path).split(".")[-1] == "txt":
      self.images_path, self.transcriptions, self.embeds_path = self.load_gt_txt(gt_file_path)
    self.nSamples_real = min(len(self.images_path), num_samples)
  def __len__(self):
    return self.nSamples_real

  # crop later
  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    # index += 1
    img_path = self.images_path[index]
    embed_path = self.embeds_path[index]
    word = self.transcriptions[index]
    try:
      img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
      if embed_path is not None:
        embed_vector = np.load(os.path.join(self.embed_path, embed_path))
      else:
        embed_vector = np.zeros(300)
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]

    if self.lowercase:
      word = word.lowercase()

    ## fill with the padding token
    label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
    label_list = []
    for char in word:
      if char in self.char2id:
        label_list.append(self.char2id[char])
      else:
        ## add the unknown token
        # print('{0} is out of vocabulary.'.format(char))
        label_list.append(self.char2id[self.UNKNOWN])
    ## add a stop token
    label_list = label_list + [self.char2id[self.EOS]]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)

    if len(label) <= 0:
      return self[index + 1]

    # label length
    label_len = len(label_list)

    if self.transform is not None:
      img = self.transform(img)
    return img, label, label_len, embed_vector

  def load_gt_json(self, gt_path):
    assert isinstance(gt_path, str), "load_gt_txt need ground truth path"
    with open(gt_path) as f:
      gt_file=json.load(f)
    images_path = []
    transcriptions = []
    embeds = []
    for k in gt_file.keys():
      annotation = gt_file[k]
      """
      if annotation['illegibility'] == True or annotation['laguage'] != 'Latin':
        continue
      """
      # images_path.append(os.path.join(self.root, k))
      images_path.append(k)
      transcriptions.append(annotation['transcription'])
      if self.embed_path is None:
        embeds.append(None)
      else:
        # embed_file_path = os.path.join(self.embed_path, k.replace("jpg", "npy"))
        embed_file_path = k.replace("jpg", "npy")
        if not os.path.exists(os.path.join(self.embed_path, k.replace("jpg", "npy"))):
          embed_file_path = k.split("/")[5] + "/" + k.split("/")[6].replace("jpg", "npy")
        # embeds.append(os.path.join(self.embed_path, k.replace("jpg", "npy")))
        embeds.append(embed_file_path)
    return images_path, transcriptions, embeds

  def load_gt_txt(self, gt_path):
    assert isinstance(gt_path, str), "load_gt_txt need ground truth path"
    images_path = []
    transcriptions = []
    embeds = []
    with open(gt_path, "r", encoding="utf-8") as f:
      for line in f.readlines():
        line = line.strip()
        line = line.split()
        if len(line) != 2:
          continue
        # images_path.append(os.path.join(self.root, line[0]))
        images_path.append(line[0])
        transcriptions.append(line[1])
        if self.embed_path is None:
          embeds.append(None)
        else:
          # embeds.append(os.path.join(self.embed_path, os.path.basename(line[0]).replace("jpg", "npy")))
          # embeds.append(os.path.join(self.embed_path, line[0].replace("jpg", "npy")))
          if "jpg" in line[0]:
            embeds.append(line[0].replace("jpg", "npy"))
          elif "png" in line[0]:
            embeds.append(line[0].replace("png", "npy"))
    return images_path, transcriptions, embeds

class LmdbDataset(data.Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform=None):
    super(LmdbDataset, self).__init__()

    if global_args.run_on_remote:
      dataset_name = os.path.basename(root)
      data_cache_url = "/cache/%s" % dataset_name
      if not os.path.exists(data_cache_url):
        os.makedirs(data_cache_url)
      if mox.file.exists(root):
        mox.file.copy_parallel(root, data_cache_url)
      else:
        raise ValueError("%s not exists!" % root)
      
      self.env = lmdb.open(data_cache_url, max_readers=32, readonly=True)
    else:
      self.env = lmdb.open(root, max_readers=32, readonly=True)

    assert self.env is not None, "cannot create lmdb from %s" % root
    self.txn = self.env.begin()

    self.voc_type = voc_type
    self.transform = transform
    self.max_len = max_len
    self.nSamples = int(self.txn.get(b"num-samples"))
    self.nSamples = min(self.nSamples, num_samples)

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)
    self.lowercase = (voc_type == 'LOWERCASE')

  def __len__(self):
    return self.nSamples

  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
      # img = Image.open(buf).convert('L')
      # img = img.convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]

    # reconition labels
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.lowercase:
      word = word.lower()
    ## fill with the padding token
    label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
    label_list = []
    for char in word:
      if char in self.char2id:
        label_list.append(self.char2id[char])
      else:
        ## add the unknown token
        # print('{0} is out of vocabulary.'.format(char))
        label_list.append(self.char2id[self.UNKNOWN])
    ## add a stop token
    label_list = label_list + [self.char2id[self.EOS]]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)

    if len(label) <= 0:
      return self[index + 1]

    # label length
    label_len = len(label_list)

    # Embedding vectors
    embed_key = b'embed-%09d' % index
    embed_vec = self.txn.get(embed_key)
    if embed_vec is not None:
      embed_vec = embed_vec.decode()
    else:
      embed_vec = ' '.join(['0']*300)
    # make string vector to numpy ndarray
    embed_vec = np.array(embed_vec.split()).astype(np.float32)
    if embed_vec.shape[0] != 300:
      return self[index + 1]
    if self.transform is not None:
      img = self.transform(img)
    return img, label, label_len, embed_vec


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img


class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())


class AlignCollate(object):

  def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    self.imgH = imgH
    self.imgW = imgW
    self.keep_ratio = keep_ratio
    self.min_ratio = min_ratio

  def __call__(self, batch):
    images, labels, lengths, embeds = zip(*batch)
    b_lengths = torch.IntTensor(lengths)
    b_labels = torch.IntTensor(labels)
    b_embeds = torch.FloatTensor(embeds)

    imgH = self.imgH
    imgW = self.imgW
    if self.keep_ratio:
      ratios = []
      for image in images:
        w, h = image.size
        ratios.append(w / float(h))
      ratios.sort()
      max_ratio = ratios[-1]
      imgW = int(np.floor(max_ratio * imgH))
      imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
      imgW = min(imgW, 400)

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)

    return b_images, b_labels, b_lengths, b_embeds

def debug():
  img_root_dir = "/data2/data/ART/train_images/"
  gt_file_path = "/data2/data/ART/train_labels.json"
  train_dataset = CustomDataset(root=img_root_dir, gt_file_path=gt_file_path, voc_type="ALLCASES_SYMBOLS", max_len=50, num_samples=5000)
  batch_size = 4
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    collate_fn=AlignCollate(imgH=64, imgW=256, keep_ratio=False))
  for i, (images, labels, lengths, masks) in enumerate(train_dataloader):
    print(i)
    # images = images.permute(0, 2, 3, 1)
    # images = to_numpy(images)
    # images = images * 0.5 + 0.5
    # images = images * 255
    # for id, (image, label, label_len) in enumerate(zip(images, labels, lengths)):
    #   image = Image.fromarray(np.uint8(image))
    #   trans = labels2strs(label, train_dataset.id2char, train_dataset.char2id)[0]
      # image.save("show_crop/" + trans + "_" + str(i) + ".jpg")
      # image = toPILImage(image)
      # image.show()
      #       # print(image.size)
      # print(labels2strs(label, train_dataset.id2char, train_dataset.char2id))
      # print(label_len.item())
      # input()
    # if i == 4:
    #   break
if __name__ == "__main__":
    debug()