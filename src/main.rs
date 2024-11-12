use burn::tensor::{Tensor};
use burn::backend::Wgpu;
use burn::prelude::*;
mod model;

type Backend = Wgpu<f32, i32>;
use burn::record::{FullPrecisionSettings, BinFileRecorder};

fn main() {
    let device = Default::default();
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);
    let my_model = model::ModelConfig::new().init::<Backend>(&device);
    let dummy_input = Tensor::<Backend, 3>::from_data([[[2., 3., 4.], [3., 5., 5.], [3.,0., 2.]]], &device);

    dbg!(&my_model);
    println!("{}", my_model.forward(dummy_input));

    println!("{}", tensor_1 + tensor_2);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    my_model.save_file("model", &recorder)
        .expect("Failed to save the model");
}
