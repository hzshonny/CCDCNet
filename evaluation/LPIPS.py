import os
import torch
from PIL import Image
import lpips
from torchvision import transforms

# 文件夹路径
folder_path1 = './output/real'  # 第一组图片所在的文件夹
folder_path2 = './output/fake' # 第二组图片所在的文件夹

# 预处理：将图片缩放到相同的尺寸并归一化到[0, 1]
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 假设你想将图片缩放到256x256
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用ImageNet的均值和标准差
])

# 初始化LPIPS损失函数
loss_fn = lpips.LPIPS(net='alex')

# 加载并处理第一组图片
images1 = []
for filename in os.listdir(folder_path1):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):  # 检查文件扩展名
        image_path = os.path.join(folder_path1, filename)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        images1.append(image_tensor)
tensors1 = torch.cat(images1, dim=0)

# 加载并处理第二组图片
images2 = []
for filename in os.listdir(folder_path2):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):  # 检查文件扩展名
        image_path = os.path.join(folder_path2, filename)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        images2.append(image_tensor)
tensors2 = torch.cat(images2, dim=0)

# 确保tensors1和tensors2在相同的设备上（如CPU或GPU）
if tensors1.device != tensors2.device:
    tensors2 = tensors2.to(tensors1.device)
# 计算LPIPS距离
lpips_distances = loss_fn(tensors1, tensors2)

# 如果tensors1和tensors2中的图片数量不一致，你可能需要调整计算方式
# 例如，只计算前min(len(tensors1), len(tensors2))个图片的距离

# 输出LPIPS距离
lpips_distances_=torch.mean(lpips_distances)
print(lpips_distances_)
# dist_=[]
# for i in range(len(tensors1)):
#     dist=loss_fn.forward(tensors1[i], tensors2[i])
#     dist_.append(dist.mean().item())
# print('Avarage Distances: %.4f' % (sum(dist_)/len(tensors1)))

# for i in range(len(images1)):
#     dummy_img0 = lpips.im2tensor(lpips.load_image(images1[i]))
#     dummy_img1 = lpips.im2tensor(lpips.load_image(images2[i]))
#     if (use_gpu):
#         dummy_img0 = dummy_img0.cuda()
#         dummy_img1 = dummy_img1.cuda()
#     dist = loss_fn.forward(dummy_img0, dummy_img1)
#     dist_.append(dist.mean().item())
# print('Avarage Distances: %.4f' % (sum(dist_) / len(images1)))
