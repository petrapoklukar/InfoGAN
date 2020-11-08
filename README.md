# InfoGAN

This is a Pytorch implementation of various versions of the InfoGAN model introduced by Chen et al. (https://arxiv.org/pdf/1606.03657.pdf). The models are structured in the following way:

- InfoGAN [mother class]
  - InfoGAN_general [original implementation of InfoGAN]
  - InfoGAN_continuous [an implementation of InfoGAN without the discrete structured latent code]
  - InfoGAN_yumi [an implementation of InfoGAN without the discrete structured latent code adjusted for training on the robot trajectories]

To train InfoGAN_general or InfoGAN_continuous:
- prepare a config file (see config/MNIST folder for samples).
- adjust ImageDataset class in train_InfoGAN_general.py or train_InfoGAN_continuous.py for image dataset of your choice. Currently only MNIST is implemented. Alternatively, adjust the class for data type of your choice.
- run the train file.

To train InfoGAN_yumi:
- prepare a config file (see config/yumi folder for samples).
- adjust TrajDataset class in train_InfoGAN_yumi.py for your robot trajectories.
- run the train file.

  
  
  
