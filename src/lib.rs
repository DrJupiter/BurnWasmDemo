use wasm_bindgen::prelude::*;
mod model;
mod state;
use burn::prelude::*;


use js_sys::Array;

type Backend = burn::backend::Wgpu;

#[wasm_bindgen]
extern "C" {
    // Javascript bindings we want to call from rust.
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(number: Vec<f32>) -> f32 {
    let sum = number.into_iter().sum::<f32>();
    // alert(&*format!("{}", &sum));
    return sum
}

#[wasm_bindgen]
pub struct MyModel {
    model: Option<model::Model<Backend>>,
}

#[wasm_bindgen]
impl MyModel {
    
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model : None }
    }

    pub async fn inference(&mut self, input: &[f32]) -> Array {
        if self.model.is_none() {
        // there should be a match statement here for multibackend support.
            self.model = state::build().await;
        }
    let device = Default::default();
    let model = self.model.as_ref().unwrap();
    let dummy = Tensor::<Backend, 3>::from_floats([[[2., 3., 4.], [3., 5., 5.], [3.,0., 2.]]], &device);
    let output = model.forward(dummy);

    let dummy = Tensor::<Backend, 3>::from_data([[[2., 3., 4.], [3., 5., 5.], [3.,0., 2.]]], &device);
    let data = dummy.into_data_async().await;
    let array = Array::new();
    for batch in data.iter::<f32>() {
        array.push(&batch.into());
        }
         
    
    return array 
    

    }
    #[wasm_bindgen]
    pub fn hi(self) {
    alert("call successful");
    }


}

