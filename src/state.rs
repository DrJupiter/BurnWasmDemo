use burn::backend::wgpu::{AutoGraphicsApi,  WgpuDevice};
use burn::backend::wgpu::init_async;
use burn::prelude::*;
use burn::module::{Module, ModuleVisitor, Param, Parameter};
use burn::tensor::Tensor;
use burn::backend::wgpu::{Wgpu};
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder};
use crate::model;
pub type Backend = Wgpu<f32, i32>;

pub async fn build() -> Option<model::Model<Backend>> {
    init_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::default().load("model.mpk".into(), &Default::default()) 
    .expect("Failed to load model");
    let m = model::ModelConfig::new().init::<burn::backend::Wgpu>(&Default::default())
    .load_record(record);
    

    let dummy_input = Tensor::<burn::backend::Wgpu, 3>::from_data([[[2., 3., 4.], [3., 5., 5.], [3.,0., 2.]]], &Default::default());
    m.forward(dummy_input);
    return Some(m)
}

