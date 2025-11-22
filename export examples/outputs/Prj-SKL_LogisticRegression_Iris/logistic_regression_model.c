#include "logisticRegression.h"
#include "logistic_regression_model.h"
#include "common.h"
#include "neural_net.h"

// Initialization function prototypes
normalization_layer_t init_Standard_Scaler_data(void);
logistic_regression_layer_t init_Logistic_Regression_data(void);


// Global Variables
normalization_layer_t Standard_Scaler_data;
logistic_regression_layer_t Logistic_Regression_data;


void model_init(){
    Standard_Scaler_data = init_Standard_Scaler_data();
    Logistic_Regression_data = init_Logistic_Regression_data();

}

void model_predict(data1d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: Standard_Scaler
    data1d_t output0;
    standard_norm_layer(Standard_Scaler_data, input, &output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: Logistic_Regression
    input = output0;
    logistic_regression_layer(Logistic_Regression_data,input, &output0);
    

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
 
static float Logistic_Regression_weights[]={ -1.003166,1.144873,-1.811348,-1.692510,0.527990,-0.283200,-0.340607,-0.720140,0.475175,-0.861673,2.151955,2.412650 };;
static float Logistic_Regression_bias[]={ -0.133772,1.982646,-1.848874 };
static float Logistic_Regression_classes[] ={ 0.000000,1.000000,2.000000 };

logistic_regression_layer_t init_Logistic_Regression_data(void)
{
    logistic_regression_layer_t lr = {
        .n_features= 4,
        .n_classes= 3,
        .weights= Logistic_Regression_weights,
        .bias= Logistic_Regression_bias,
        .classes = Logistic_Regression_classes,
    };
    return lr;
}

