1、每次运行不同任务时需要修改Main.py中的“state":
	train_diffusion:训练diffusion moel
	sample:采样一小块
	train_cls:训练分类网络
	generate_img:使用PAN和MS生成一张大图

2、train_diffusion时：不需要动Main.py中一些东西，除过超参和模型路径, 需要修改Ms4_patch_size为64

3、sample：需要修改Main.py中的image_size=512，因为diffusion model是支持512x512的图片。还可以修改Main.py中的”timestep_respacing":[]，这是采样的步数。
但是”timestep_respacing"应小于“diffusion_steps”。

4、train_cls：修改Main.py中的image_size=64，因为这符合切片大小，其中还可以修改一些超参数。在Scripts/train_cls.py中，需要根据不同的数据集修改model(output=该数据集类别的数目)，且在test中修改混淆矩阵test_matrix=torch.zeros([类别数，类别数])。

5、generate_img：需要修改Main.py中的image_size=512，因为diffusion model是支持512x512的图片。还可以修改Main.py中的”timestep_respacing":[]，这是采样的步数。
但是”timestep_respacing"应小于“diffusion_steps”。在Scripts/save_fusion_image.py
中修改MS和PAN的路径和生成新图的名字。

6、label_convert：将.mat文件转换为.npy

7、Scheduler：学习率的调配算法。

8、Diffusion/gaussian_diffusion：最主要的diffusion部分代码

9、Diffusion/cls_model：分类网络

10、在进行上色的时候需要修改的有color.py里的color_count, out_color以及上色的RGB值,还有模型的输出model=cls_model(output=?)

11、在进行模块测试的时候，还修改了Diffusion/gaussian_diffusion中的函数ddim loop中的一些部分。




