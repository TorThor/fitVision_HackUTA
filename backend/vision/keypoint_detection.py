import kagglehub
import os

# Download latest version
movenet_dir = os.path.join(os.getcwd(), 'backend\MoveNet_model')
# print("MoveNet dir:", movenet_dir)
os.chdir(movenet_dir)
path = kagglehub.model_download("google/movenet/tfLite/singlepose-thunder")

