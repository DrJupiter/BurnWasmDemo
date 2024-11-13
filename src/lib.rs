use wasm_bindgen::prelude::*;
mod model;
// mod state;
use burn::prelude::*;
use model::{ModelConfig, Model};

use burn::backend::wgpu::{init_async, AutoGraphicsApi, Wgpu, WgpuDevice};




#[wasm_bindgen]
extern "C" {
    // Javascript bindings we want to call from rust.
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(number: Vec<f32>) -> f32 {
    let sum = number.into_iter().sum::<f32>();
    return sum
}

#[wasm_bindgen]
pub struct MyModel {
    model: model::Model<Wgpu<f32, i32>>,
    device: WgpuDevice,
}

#[wasm_bindgen]
impl MyModel {
    
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Self {
        let device = WgpuDevice::default();
        init_async::<AutoGraphicsApi>(&device, Default::default()).await;
        Self { model : ModelConfig::new().init(&device), device: device }
    }

    pub async fn inference(&mut self, input: &[f32]) -> Vec<f32> {
    // let model = self.model.as_ref().unwrap();
    let input = Tensor::<Wgpu<f32, i32>, 1>::from_floats(input, &self.device).reshape([1,3,3]);
    let dummy = Tensor::<Wgpu<f32, i32>, 3>::from_data([[[2., 3., 4.], [3., 5., 5.], [3.,0., 2.]]], &self.device);
    let output = self.model.forward(input);
    let output = output.into_data_async().await;
    let output = output.convert::<f32>().to_vec().unwrap();

    return output;
         
    
    

    }
    #[wasm_bindgen]
    pub fn hi(self) {
    alert("call successful");
    }


}

