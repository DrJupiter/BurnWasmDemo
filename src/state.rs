use burn::{
    backend::wgpu::{self, AutoGraphicsApi, Wgpu, WgpuDevice, init_async},
    module::{Module, ModuleVisitor, Param, Parameter},
    prelude::*,
    tensor::Tensor,
    record::{BinBytesRecorder,FullPrecisionSettings, Recorder},
};
use crate::model;


static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

pub type Backend = Wgpu<f32, i32>;

pub async fn build() -> model::Model<Backend> {

    init_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    let m: model::Model<Backend> = model::ModelConfig::new().init::<burn::backend::Wgpu>(&Default::default());
    

    let record = BinBytesRecorder::<FullPrecisionSettings>::default().load(STATE_ENCODED.to_vec(), &Default::default())
        .expect("Failed to decode state");

    m.load_record(record)
}

