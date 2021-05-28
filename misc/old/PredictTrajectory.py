import matplotlib.pyplot as plt
from nuscenes.prediction.models.mtp import MTP
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import CoverNet
from emptyMap import emptyStaticLayerRasterizer
import torch
import numpy as np
import pickle
np.random.seed(123)
torch.manual_seed(123)
DATAROOT = 'PATH/TO/nuscenes'

"""
code for using nuscenes dataset. If you have tha input image to covernet you can
skip loading in images from nuscense
"""
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
helper = PredictHelper(nuscenes)

static_layer_rasterizer = StaticLayerRasterizer(helper)
static_layer_rasterizer_e = emptyStaticLayerRasterizer()
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)
#img = "load numpy array of image"

backbone = ResNetBackbone('resnet50')
covernet = CoverNet(backbone, num_modes=64)

# agent data that needs to be pulled from ego vehicle 
agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                    helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                    helper.get_heading_change_rate_for_agent(instance_token_img, sample_token_img)]])
image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

logits = covernet(image_tensor, agent_state_vector)
print(logits)


# Epsilon is the amount of coverage in the set, 
# i.e. a real world trajectory is at most 8 meters from a trajectory in this set
# We released the set for epsilon = 2, 4, 8. Consult the paper for more information
# on how this set was created

PATH_TO_EPSILON_8_SET = "/scratch/deonrich/trajectories/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
trajectories = pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb'))

# Saved them as a list of lists
trajectories = torch.Tensor(trajectories)

# Print 5 most likely predictions
traject = trajectories[logits.argsort(descending=True)[0,:5]]
trajectImg = static_layer_rasterizer_e.plotTrajectories(traject)
plt.imshow(trajectImg)
plt.savefig("TrajectoryImg.png")
