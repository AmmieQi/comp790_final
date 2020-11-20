# Course Project for Comp790

To run digits experiment:

1. ``` cd digits ```
2. Pretrain digit: ``` train_mnist.sh ``` or ``` train_svhn.sh ``` or ``` train_usps.sh ```
3. Run Adaptation: ``` bash usps2mnist.sh ``` if you want to run ```usps``` to ```mnist```, for example.

To run DomainNet experiment:

1. ``` cd domainnet ```
2. Pretrain DomainNet: ``` train_clipart.sh ``` or ``` train_real.sh ``` or ``` train_sketch.sh ```
3. Run Adaptation: ``` bash sketch2clipart.sh ``` if you want to run ```sketch``` to ```clipart```, for example.
