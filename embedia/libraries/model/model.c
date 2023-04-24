{includes}
// Initialization function prototypes
{prototypes_init}

// Global Variables
{var}

void model_init(){{
{init}
}}

void model_predict({input_data_type} input, {output_data_type} * output){{
  
{predict}

    *output = {output_name};

}}

int model_predict_class({input_data_type} input, {output_data_type} * results){{
  
   
    model_predict(input, results);
    
    {predict_class}
    //return argmax(data1d_t);

}}

// Implementation of initialization functions

{functions_init}
