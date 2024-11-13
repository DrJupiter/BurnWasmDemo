use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1,8], [3,3]).init(device),
        }

    }

}

impl<B: Backend> Model<B> {

pub fn forward(&self, data: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch_size, height, width] = data.dims();

    let x = data.unsqueeze::<4>();
    let y = self.conv1.forward(x.clone());

    return x

}

}
