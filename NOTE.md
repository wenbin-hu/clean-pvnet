## 安装相关

### CUDA相关
- clean-pvnet要求CUDA9.0，然而英伟达官网CUDA9.0下载链接挂了，故安装CUDA10.0
- 跟着https://blog.csdn.net/weixin_44120025/article/details/121002696做，该博客讲如何维护/切换多版本CUDA
- 在https://developer.nvidia.cn/rdp/cudnn-archive下载cuDNN，注意路径中的.cn，一般教程都会用.com，但很多时候.com很慢或打不开
- 在安装好CUDA，配置好环境变量后，如果nvcc --version显示nvcc未安装，应尝试重启，根据https://github.com/akirademoss/cuda-9.0-installation-on-ubuntu-18.04
- 如果报错ImportError: libcudart.so.10.0: cannot open shared object file: No such file or directory，解决方法sudo ldconfig /usr/local/cuda-10.0/lib64。参考https://blog.csdn.net/c2250645962/article/details/105671162

### 其他
- 如果报错：ImportError: cannot import name ‘PILLOW_VERSION‘ from ‘PIL‘
把相关文件中的PILLOW_VERSION改成_version。原因是Pillow7.0.0之后就不用PILLOW_VERSION了。参考：https://www.pudn.com/news/623e0595f9448f597f2900ae.html

- 需要安装transforms3d: pip3 install transforms3d

- 跑detector需要安装imgaug: pip3 install imgaug

- 报错 RuntimeError: cublas runtime error : the GPU program failed to execute at /pytorch/aten/src/THC/THCBlas.cu:450
可能原因：CUDA和torch版本不对应，clean-pvnet要求CUDA9.0，我安装的是CUDA10.0，需重新安装对应版本torch
根据https://pytorch.org/get-started/previous-versions/，执行conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch


- 训练时报错：
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
  1. Downgrade the protobuf package to 3.20.x or lower.
  2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
  
More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
解决： 将protobuf 4.21.1版本降级到3.20.x，通常即可解决问题。 pip3 install --upgrade protobuf==3.20.1
参考：https://www.cnblogs.com/booturbo/p/16339195.html

- 使用ObjectDatasetTools制作数据集时报错：
scipy.spatial.qhull.QhullError: QH7023 qhull option warning: unknown 'Q' qhull option 'Qn', skip to next space
QH6035 qhull option error: see previous warnings, use 'Qw' to override: 'qhull i Qt QJn Pp QbB' (last offset 12)
While executing:  | qhull i Qt QJn Pp QbB
Options selected for Qhull 2019.1.r 2019/06/21:
  run-id 34560705  incidence  Qtriangulate  Pprecision-ignore
  QbBound-unit-box 0.5  _maxoutside  0

解决方法：降低scipy版本：
pip uninstall scipy
pip install scipy==1.3.3
参考：https://blog.csdn.net/onepunch_k/article/details/127167577   https://github.com/mikedh/trimesh/issues/670

------------------------------------------------------------------------------------------------
## 训练笔记
### 数据集准备相关，使用ObjectDatasetTools (即pvnet作者自己写的工具)
- 使用realsenseD455相机之前，应对其进行标定(calibration)，可以使用realsense-viewer里的On-chip calibration按钮（最好打印一张target, 参考https://www.intelrealsense.com/self-calibration-for-depth-cameras/）。也可以使用D400 Series Calibration Tools（https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_D400_Dyn_Calib_User_Guide.pdf）
- 调用record2.py程序拍摄训练图片时，距离不能太近，否则无法得到有效的深度图，D455的minZ是35cm(大概)
- 图片数量不能太多，600张比较合适，否则执行compute_gt_poses.py时，内存不足。大概是因为储存pose_graph。
- 拍摄的时候不要把不相关的背景拍出来，也不要拍到移动的物体（比如人），否则重建出来的点云图有重影（两个不同的图叠在一起）
- 得到重建好的图之后，使用MeshLab处理
	- 把背景删掉，只留下物体
	- 创建normals(Filters, Normals..., Compute normals from point sets)
	- 创建mesh(Filters, Remashing..., Surface reconstruction: Screened Possion)
	- 手动删掉不合适的部分
	- 把底部填平(Filters, Remashing..., Close Holes)
	- 保存；保存的时候要把faces勾选上，然后pvnet要求保存为ascii格式，故不要勾选Binary 	Encoding(Gen6D好像要求binary格式)
- 得到masks, poses后，需要处理一下pose文件夹下的poseX.npy。因为ObjectDatasetTools生成的poseX.npy存的是4x4的矩阵，但pvnet需要3x4的矩阵，故需要把最后一行0 0 0 1删掉。
- 根据intrinsics.json创建camera.txt。矩阵为[fx, 0, ppx; 0, fy, ppy; 0, 0, 1]
- diameter.txt中存物体的实际大小，只有一个数字，以米为单位，我认为是物体bounding box最长边的长度。
	
### 对pvnet的使用理解
- 首先使用ObjectDatasetTools生成训练集，得到rgb, mask和相机pose。然后用run.py --type custom进行后处理，生成train.json。后处理主要分为
两步，第一步根据物体的mesh， 使用Farthest Point Sampling(FPS)得到3D特征点，第二步根据每张图片相机的pose，得到2D图片上的特征点。
- 网络的输入：rgb图片
- 已知的信息：物体的3D模型，3D模型下的关键点、bounding box、中心点坐标
- 网络的输出：'seg', 'vertex', 'mask', 'kpt_2d'
- 使用<code>python run.py --type custom</code>准备数据集，这行指令干了两件事：
  - 根据物体的mesh(model.ply)，使用Farthest point sampling(FPS)算法生成关键点的3D相对坐标，并存储在fps.txt文件下
  - 生成训练数据，例如对于600张rgb图片，已知的通用的信息为：相机内参矩阵K，3D物体bounding box坐标，3D物体中心坐标，3D物体关键点坐标，数据路径；每张
图片单独的标注信息：2D物体BB坐标，2D物体关键点坐标，2D物体中心坐标。

### 2022-12-02
训练的到的model(训练数据放在data/mug_no_augment，模型放在data/model/mug_no_augment下)，在训练集下表现优异，在测试集上几乎无法正确生成
mask和特征点。猜测为过拟合，故替换背景重新训练：12张背景图，每张杯子图片根据mask随机替换5张背景，600*(1+5)=3600张图片。
但效果依旧很差（相对来说要好一点点），没能生成正确的mask（有一半图片没有任何mask，另一半有一点点mask）和关键点。

我用相同背景的测试集（即杯子放在标定板上）测试模型，效果很好，说明问题大概率是过拟合。

### 2022-12-03
和1202的策略一样，使用背景替换。改进之处是杯子的mask更加紧凑，更贴合杯子的边缘（通过改ObjectDatasetTools/create_label_files.py），同时我自
己拍了几张背景。效果在测试集上依然很差，今天尝试旋转、拉伸（放缩？）、平移的操作。

### 2022-12-05
600张原始训练图片(存放在<code>data/original_mug</code>)，每一张都使用<code>tools/data_augmentation.py</code>经过旋转、平移和缩放
（无截断），得到600张新的图片。
- angle = np.random.randint(low=-60, high=60)
- scale = np.random.uniform(low=0.5, high=1.5)
- dx = np.random.randint(low=-200, high=200)
- dy = np.random.randint(low=-200, high=200)

之后再使用<code>background_augmentation.py</code>加上随机背景替换，得到1200张图片，拿来当训练集。得到的效果是目前为止最好的，在train, 
close-to-far, cluttered上有百分百成功率，在far上有80%左右成功率，在close上有大概50%成功率，在blend上完全不行。可以尝试通过增大缩放的幅度（更大和更小）来提升性能。

### 2022-12-06
两个改动：1.增加两张实验室桌子的背景 2.放缩的尺度更大(low=0.25, high=2.0)。其他和12-05一样，共1200张图片。
似乎效果有些微的提升，需要用指标去定量的分析（在测试集上没有ground truth咋整）。