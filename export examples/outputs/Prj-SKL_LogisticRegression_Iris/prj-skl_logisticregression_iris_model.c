#include "logisticRegression.h"
#include "common.h"
#include "prj-skl_logisticregression_iris_model.h"
#include "neural_net.h"

// Initialization function prototypes
normalization_layer_t init_Standard_Scaler_data(void);


// Global Variables
normalization_layer_t Standard_Scaler_data;


void model_init(){
    Standard_Scaler_data = init_Standard_Scaler_data();

}

void model_predict(data1d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: Standard_Scaler
    data1d_t output0;
    standard_norm_layer(Standard_Scaler_data, input, &output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: Prj-SKL_Logistic_Regression_Iris
    input = output0;
    
    
    static logistic_regression_layer_t lr;
    static int initialized = 0; // Bandera
    
    if (Prj-SKL_Logistic_Regression_Iris_initialized == 0) {
            Prj-SKL_Logistic_Regression_Iris_layer = init_Prj-SKL_Logistic_Regression_Iris_data();
            Prj-SKL_Logistic_Regression_Iris_initialized= 1;
    }
    
        logistic_regression_layer(Prj-SKL_Logistic_Regression_Iris_layer,input, output0;
    
    

    *output = output0;

}

int model_predict_class(data1d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return results->data[0];
    //return argmax(data1d_t);

}

// Implementation of initialization functions


normalization_layer_t init_Standard_Scaler_data(void){
    /*[5.80916667 3.06166667 3.72666667 1.18333333]*/
    static const float sub_val[] ={
    5.809166666666665, 3.0616666666666674, 3.726666666666667, 1.1833333333333333
    };
    /*[1.21896909 2.23589719 0.57305675 1.33485032]*/
    static const float inv_div_val[] ={
    1.2189690866760947, 2.2358971863210813, 0.5730567543113094, 1.3348503216138696, 
  
    };

    static const normalization_layer_t norm = { sub_val, inv_div_val  };
    return norm;
}

