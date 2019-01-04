import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

class Pose_Estimation(nn.Module):

    def __init__(self,num_vertices, num_vector,batch_norm=False):
        
        super(Pose_Estimation, self).__init__()

	# PRETRAINED VGG-LAYERS
	self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
	self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
	self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
	self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
	self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
	self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
	self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
	self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
	self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

	self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
	self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
	self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

	#STAGE 1	
        self.conv1_ver = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_ver = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_ver = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_ver = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_ver = nn.Conv2d(in_channels=512, out_channels=num_vertices, kernel_size=1, stride=1, padding=0)

        self.conv1_vec = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_vec = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
	self.conv3_vec = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
	self.conv4_vec = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
	self.conv5_vec = nn.Conv2d(in_channels=512, out_channels=num_vector*2, kernel_size=1, stride=1, padding=0)

	#STAGE 2
	self.conv1_ver2 = nn.Conv2d(in_channels=128+num_vertices+num_vector*2, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv2_ver2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv3_ver2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv4_ver2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv5_ver2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv6_ver2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
	self.conv7_ver2 = nn.Conv2d(in_channels=128, out_channels=num_vertices, kernel_size=1, stride=1, padding=0)
	
	self.conv1_vec2 = nn.Conv2d(in_channels=128+num_vertices+num_vector*2, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv2_vec2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv3_vec2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv4_vec2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv5_vec2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv6_vec2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
	self.conv7_vec2 = nn.Conv2d(in_channels=128, out_channels=num_vector*2, kernel_size=1, stride=1, padding=0)

	#STAGE 3
	self.conv1_ver3 = nn.Conv2d(in_channels=128+num_vertices+num_vector*2, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv2_ver3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv3_ver3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv4_ver3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv5_ver3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv6_ver3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
	self.conv7_ver3 = nn.Conv2d(in_channels=128, out_channels=num_vertices, kernel_size=1, stride=1, padding=0)
	
	self.conv1_vec3 = nn.Conv2d(in_channels=128+num_vertices+num_vector*2, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv2_vec3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv3_vec3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv4_vec3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv5_vec3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv6_vec3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
	self.conv7_vec3 = nn.Conv2d(in_channels=128, out_channels=num_vector*2, kernel_size=1, stride=1, padding=0)

	#STAGE 4
	self.conv1_ver4 = nn.Conv2d(in_channels=128+num_vertices+num_vector*2, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv2_ver4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv3_ver4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv4_ver4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv5_ver4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv6_ver4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
	self.conv7_ver4 = nn.Conv2d(in_channels=128, out_channels=num_vertices, kernel_size=1, stride=1, padding=0)

	self.conv1_vec4 = nn.Conv2d(in_channels=128+num_vertices+num_vector*2, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv2_vec4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv3_vec4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv4_vec4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv5_vec4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
	self.conv6_vec4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
	self.conv7_vec4 = nn.Conv2d(in_channels=128, out_channels=num_vector*2, kernel_size=1, stride=1, padding=0)



    def forward(self, x, mask):
	
	#x = F.relu(self.conv1_1(x))
	#x = self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
	#x = F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))))
        #x = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x))))))))))
	#x = F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x))))))))))))
        #x = F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x))))))))))))))
	#x = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x))))))))))))))))
	#x = self.pool3(F.relu(self.conv3_4(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))))))))))))))))
	#x = F.relu(self.conv4_1(self.pool3(F.relu(self.conv3_4(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))))))))))))))))))
	#x = F.relu(self.conv4_2(F.relu(self.conv4_1(self.pool3(F.relu(self.conv3_4(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))))))))))))))))))))
	#x = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(self.pool3(F.relu(self.conv3_4(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))))))))))))))))))))))
	x = F.relu(self.conv4_4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(self.pool3(F.relu(self.conv3_4(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))))))))))))))))))))))))

	#print "x: "
	#print x.size()
        #out_ver = F.relu(self.conv1_ver(x))
        #out_ver = F.relu(self.conv2_ver(F.relu(self.conv1_ver(x))))
        #out_ver = F.relu(self.conv3_ver(F.relu(self.conv2_ver(F.relu(self.conv1_ver(x))))))
        #out_ver = F.relu(self.conv4_ver(F.relu(self.conv3_ver(F.relu(self.conv2_ver(F.relu(self.conv1_ver(x))))))))
        out_ver = self.conv5_ver(F.relu(self.conv4_ver(F.relu(self.conv3_ver(F.relu(self.conv2_ver(F.relu(self.conv1_ver(x)))))))))
	#print "outver: "
	#print out_ver.size()
        #out_vec = F.relu(self.conv1_vec(x))
        #out_vec = F.relu(self.conv2_vec(F.relu(self.conv1_vec(x))))
        #out_vec = F.relu(self.conv3_vec(F.relu(self.conv2_vec(F.relu(self.conv1_vec(x))))))
        #out_vec = F.relu(self.conv4_vec(F.relu(self.conv3_vec(F.relu(self.conv2_vec(F.relu(self.conv1_vec(x))))))))
        out_vec = self.conv5_vec(F.relu(self.conv4_vec(F.relu(self.conv3_vec(F.relu(self.conv2_vec(F.relu(self.conv1_vec(x)))))))))
	#print "outvec: "
	#print out_vec.size()
	out1 = torch.cat([out_ver,out_vec,x],1)
        out_ver_mask = out_ver*mask
        out_vec_mask = out_vec*mask
	
	#print "out1: "
	#print out1.size()
        #out_ver2 = F.relu(self.conv1_ver2(out1))
        #out_ver2 = F.relu(self.conv2_ver2(F.relu(self.conv1_ver2(out1))))
        #out_ver2 = F.relu(self.conv3_ver2(F.relu(self.conv2_ver2(F.relu(self.conv1_ver2(out1))))))
        #out_ver2 = F.relu(self.conv4_ver2(F.relu(self.conv3_ver2(F.relu(self.conv2_ver2(F.relu(self.conv1_ver2(out1))))))))
        #out_ver2 = F.relu(self.conv5_ver2(F.relu(self.conv4_ver2(F.relu(self.conv3_ver2(F.relu(self.conv2_ver2(F.relu(self.conv1_ver2(out1))))))))))
	#out_ver2 = F.relu(self.conv6_ver2(F.relu(self.conv5_ver2(F.relu(self.conv4_ver2(F.relu(self.conv3_ver2(F.relu(self.conv2_ver2(F.relu(self.conv1_ver2(out1))))))))))))
	out_ver2 = self.conv7_ver2(F.relu(self.conv6_ver2(F.relu(self.conv5_ver2(F.relu(self.conv4_ver2(F.relu(self.conv3_ver2(F.relu(self.conv2_ver2(F.relu(self.conv1_ver2(out1)))))))))))))
	#print "outver2: "
	#print out_ver2.size()
        #out_vec2 = F.relu(self.conv1_vec2(out1))
        #out_vec2 = F.relu(self.conv2_vec2(F.relu(self.conv1_vec2(out1))))
        #out_vec2 = F.relu(self.conv3_vec2(F.relu(self.conv2_vec2(F.relu(self.conv1_vec2(out1))))))
        #out_vec2 = F.relu(self.conv4_vec2(F.relu(self.conv3_vec2(F.relu(self.conv2_vec2(F.relu(self.conv1_vec2(out1))))))))
        #out_vec2 = F.relu(self.conv5_vec2(F.relu(self.conv4_vec2(F.relu(self.conv3_vec2(F.relu(self.conv2_vec2(F.relu(self.conv1_vec2(out1))))))))))
	#out_vec2 = F.relu(self.conv6_vec2(F.relu(self.conv5_vec2(F.relu(self.conv4_vec2(F.relu(self.conv3_vec2(F.relu(self.conv2_vec2(F.relu(self.conv1_vec2(out1))))))))))))
	out_vec2 = self.conv7_vec2(F.relu(self.conv6_vec2(F.relu(self.conv5_vec2(F.relu(self.conv4_vec2(F.relu(self.conv3_vec2(F.relu(self.conv2_vec2(F.relu(self.conv1_vec2(out1)))))))))))))
	#print "outvec2: "
	#print out_vec2.size()

	out2 = torch.cat([out_ver2,out_vec2,x],1)
        out_ver2_mask = out_ver2*mask
        out_vec2_mask = out_vec2*mask
	
	#print "out2: "
	#print out2.size()
        #out_ver3 = F.relu(self.conv1_ver3(out2))
        #out_ver3 = F.relu(self.conv2_ver3(F.relu(self.conv1_ver3(out2))))
        #out_ver3 = F.relu(self.conv3_ver3(F.relu(self.conv2_ver3(F.relu(self.conv1_ver3(out2))))))
        #out_ver3 = F.relu(self.conv4_ver3(F.relu(self.conv3_ver3(F.relu(self.conv2_ver3(F.relu(self.conv1_ver3(out2))))))))
        #out_ver3 = F.relu(self.conv5_ver3(F.relu(self.conv4_ver3(F.relu(self.conv3_ver3(F.relu(self.conv2_ver3(F.relu(self.conv1_ver3(out2))))))))))
	#out_ver3 = F.relu(self.conv6_ver3(F.relu(self.conv5_ver3(F.relu(self.conv4_ver3(F.relu(self.conv3_ver3(F.relu(self.conv2_ver3(F.relu(self.conv1_ver3(out2))))))))))))
	out_ver3 = self.conv7_ver3(F.relu(self.conv6_ver3(F.relu(self.conv5_ver3(F.relu(self.conv4_ver3(F.relu(self.conv3_ver3(F.relu(self.conv2_ver3(F.relu(self.conv1_ver3(out2)))))))))))))
	#print "outver3: "
	#print out_ver3.size()

        #out_vec3 = F.relu(self.conv1_vec3(out2))
        #out_vec3 = F.relu(self.conv2_vec3(F.relu(self.conv1_vec3(out2))))
        #out_vec3 = F.relu(self.conv3_vec3(F.relu(self.conv2_vec3(F.relu(self.conv1_vec3(out2))))))
        #out_vec3 = F.relu(self.conv4_vec3(F.relu(self.conv3_vec3(F.relu(self.conv2_vec3(F.relu(self.conv1_vec3(out2))))))))
        #out_vec3 = F.relu(self.conv5_vec3(F.relu(self.conv4_vec3(F.relu(self.conv3_vec3(F.relu(self.conv2_vec3(F.relu(self.conv1_vec3(out2))))))))))
	#out_vec3 = F.relu(self.conv6_vec3(F.relu(self.conv5_vec3(F.relu(self.conv4_vec3(F.relu(self.conv3_vec3(F.relu(self.conv2_vec3(F.relu(self.conv1_vec3(out2))))))))))))
	out_vec3 = self.conv7_vec3(F.relu(self.conv6_vec3(F.relu(self.conv5_vec3(F.relu(self.conv4_vec3(F.relu(self.conv3_vec3(F.relu(self.conv2_vec3(F.relu(self.conv1_vec3(out2)))))))))))))
	#print "outvec3: "
	#print out_vec3.size()

	out3 = torch.cat([out_ver3,out_vec3,x],1)
        out_ver3_mask = out_ver3*mask
        out_vec3_mask = out_vec3*mask
	#print "out3: "
	#print out3.size()
        #out_ver4 = F.relu(self.conv1_ver(out3))
        #out_ver4 = F.relu(self.conv2_ver4(F.relu(self.conv1_ver(out3))))
        #out_ver4 = F.relu(self.conv3_ver4(F.relu(self.conv2_ver4(F.relu(self.conv1_ver(out3))))))
        #out_ver4 = F.relu(self.conv4_ver4(F.relu(self.conv3_ver4(F.relu(self.conv2_ver4(F.relu(self.conv1_ver(out3))))))))
        #out_ver4 = F.relu(self.conv5_ver4(F.relu(self.conv4_ver4(F.relu(self.conv3_ver4(F.relu(self.conv2_ver4(F.relu(self.conv1_ver(out3))))))))))
	#out_ver4 = F.relu(self.conv6_ver4(F.relu(self.conv5_ver4(F.relu(self.conv4_ver4(F.relu(self.conv3_ver4(F.relu(self.conv2_ver4(F.relu(self.conv1_ver(out3))))))))))))
	out_ver4 = self.conv7_ver4(F.relu(self.conv6_ver4(F.relu(self.conv5_ver4(F.relu(self.conv4_ver4(F.relu(self.conv3_ver4(F.relu(self.conv2_ver4(F.relu(self.conv1_ver4(out3)))))))))))))
	#print "outver4: "
	#print out_ver4.size()
        #out_vec4 = F.relu(self.conv1_vec4(out3))
        #out_vec4 = F.relu(self.conv2_vec4(F.relu(self.conv1_vec4(out3))))
        #out_vec4 = F.relu(self.conv3_vec4(F.relu(self.conv2_vec4(F.relu(self.conv1_vec4(out3))))))
        #out_vec4 = F.relu(self.conv4_vec4(F.relu(self.conv3_vec4(F.relu(self.conv2_vec4(F.relu(self.conv1_vec4(out3))))))))
        #out_vec4 = F.relu(self.conv5_vec4(F.relu(self.conv4_vec4(F.relu(self.conv3_vec4(F.relu(self.conv2_vec4(F.relu(self.conv1_vec4(out3))))))))))
	#out_vec4 = F.relu(self.conv6_vec4(F.relu(self.conv5_vec4(F.relu(self.conv4_vec4(F.relu(self.conv3_vec4(F.relu(self.conv2_vec4(F.relu(self.conv1_vec4(out3))))))))))))
	out_vec4 = self.conv7_vec4(F.relu(self.conv6_vec4(F.relu(self.conv5_vec4(F.relu(self.conv4_vec4(F.relu(self.conv3_vec4(F.relu(self.conv2_vec4(F.relu(self.conv1_vec4(out3)))))))))))))
	#print "outvec4: "
	#print out_vec4.size()
	#out4 = torch.cat([out_ver4,out_vec4,x],1)
        out_ver4_mask = out_ver4*mask
        out_vec4_mask = out_vec4*mask

        return out_ver_mask,out_vec_mask,out_ver2_mask,out_vec2_mask,out_ver3_mask,out_vec3_mask,out_ver4_mask,out_vec4_mask


def PoseModel(num_vertices, num_vector, batch_norm=False, pretrained=True):

    model = Pose_Estimation(num_vertices,num_vector,batch_norm)

    parameter_num = 10

    if batch_norm:
        vgg19 = models.vgg19_bn(pretrained=True)
        parameter_num *= 6
    else:
        vgg19 = models.vgg19(pretrained=True)
        parameter_num *= 2

    vgg19_state_dict = vgg19.state_dict()
    vgg19_keys = vgg19_state_dict.keys()

    model_dict = model.state_dict()
    weights_load = OrderedDict()

    for i in range(parameter_num):
        weights_load[model_dict.keys()[i]] = vgg19_state_dict[vgg19_keys[i]]
    
    model_dict.update(weights_load)
    model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':
    print(PoseModel(19, 19))
