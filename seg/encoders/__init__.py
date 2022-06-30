from seg.encoders.resnet import resnet_encoders
from seg.encoders.vgg import vgg_encoders
from seg.encoders.densenet import densenet_encoders
from seg.encoders.mobilenet import mobilenet_encoders


cfg_encoders = {}
cfg_encoders.update(resnet_encoders)
cfg_encoders.update(vgg_encoders)
cfg_encoders.update(densenet_encoders)
cfg_encoders.update(mobilenet_encoders)
