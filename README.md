# CN Translate

# Amazon SageMaker 处理 Video2frame 模型推断

本演示展示了如何使用 SageMaker 处理过程视频帧提取和模型推理。

一些业务场景需要使用机器学习来处理视频。他们通常需要从视频中提取帧，然后将它们发送到模型并获得结果。这需要您提取帧并存储在某个地方，然后使用批量转换器或在线推理，这将涉及推理后不再需要的存储成本。因此客户正在寻找一种有效完成此类工作的方法，这里我们将介绍 Amazon SageMaker 处理。

Amazon SageMaker 处理是 Amazon SageMaker 的一项新功能，可让客户在完全托管的基础设施上轻松运行预处理、后处理和模型评估工作负载，在 re:Invent 2019 期间发布。

在此示例中，我们将在 VPC 中启动 sagemaker 处理作业，输入是 S3 中的视频，输出是推理结果（分割图像）并将存储在 S3 中。

1. 启动一个 EC2 实例作为 API 服务器，可以被 sagemaker 处理作业调用。
2. 我们使用预训练模型从 GluonCV 模型动物园做语义分割推断。
3. 启用 Sagemaker Processing vpc 模式，以便它可以调用 API 服务器。

这是此示例的高级架构。

![高级架构](https://sagemaker-demo-dataset.s3-us-west-2.amazonaws.com/Picture1.png)


### GluonCV
---
[GluonCV](https://gluon-cv.mxnet.io/) 提供了计算机视觉中最先进的 (SOTA) 深度学习算法的实现。它旨在帮助工程师、研究人员和学生快速制作产品原型、验证新想法并学习计算机视觉。

【GluonCV模型动物园】（https://gluon-cv.mxnet.io/model_zoo/index.html）包含六种预训练模型：分类、目标检测、分割、姿态估计、动作识别和深度预测。

在这个示例中，我们将使用来自 Segmentation 的 **deeplab_resnet101_citys** 并使用 **[cityscape dataset](https://www.cityscapes-dataset.com/)** 进行训练，该数据集专注于对城市街景的语义理解，所以这个模型适用于汽车视图图像。

### 先决条件

---
为了下载 GPU 支持的预测试模型，您需要在 **GPU based instance** 中运行此示例，例如 **ml.p2.xlarge 或 ml.p3.2xlarge**。

如果您只在处理作业（例如 c5 类型）中启动非 gpu 实例，则可以在非基于 gpu 的实例中运行此演示。


＃＃ 安全

 有关详细信息，请参阅 [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

＃＃ 执照

 该库在 MIT-0 许可下获得许可。请参阅 [LICENSE](LICENSE) 文件。


# Amazon SageMaker Processing Video2frame Model Inference

This demo shows how to use SageMaker processing process video frames extraction and model inference.

Some business scenario need to processing videos by using machine learning. They usually need extract frames from videos and then send them to models and get the result. This need you extract the frames and store in some place and then using batch transformer or online inference, which would involve a storage cost which is no longer need after inference. So customers are looking for a way to finish such job in a effective way, here we would introduce Amazon SageMaker Processing.

Amazon SageMaker Processing, a new capability of Amazon SageMaker that lets customers easily run the preprocessing, postprocessing and model evaluation workloads on fully managed infrastructure, was announced during re:Invent 2019. 

In this sample, we would lauch a sagemaker processing job in a VPC, the input is videos in S3, and output is inference results (segmentation images) and will be stored in S3.

1. Launch an EC2 instance to play as API server which could be called by sagemaker processing job.
2. We use pretrained model to do semantic segmentation inference from GluonCV model zoo.
3. Enable Sagemaker Processing vpc mode so it could call API server.

Here is the high level architecture of this sample.

![High level architecture](https://sagemaker-demo-dataset.s3-us-west-2.amazonaws.com/Picture1.png)


### GluonCV
---
[GluonCV](https://gluon-cv.mxnet.io/) provides implementations of state-of-the-art (SOTA) deep learning algorithms in computer vision. It aims to help engineers, researchers, and students quickly prototype products, validate new ideas and learn computer vision.

[GluonCV model zoo](https://gluon-cv.mxnet.io/model_zoo/index.html) contains six kinds of pretrained model: Classification, Object Detection, Segmentation, Pose Estimation, Action Recognition and Depth Prediction.

In this sample, we will use **deeplab_resnet101_citys** from Segmentation and was trained with **[cityscape dataset](https://www.cityscapes-dataset.com/)**, which focuses on semantic understanding of urban street scenes, so this model is suitable for car view images.

### Prerequisite
---
In order to download GPU supported pretrianed model, you need run this sample in **GPU based instance**, such as **ml.p2.xlarge or ml.p3.2xlarge**.

If you only launch none gpu instances in processing jobs, such as c5 type, you could run this demo in none gpu based instances.


## Security

 See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

 This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.   




