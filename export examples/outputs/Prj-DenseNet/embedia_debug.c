#include "embedia_debug.h"
#include "embedia_debug_def.h"


const char* STR_CONTENT = "  Content:";
const char* SEP_LINE = "";
const char* STR_CHN = "MCh: ";
const char* STR_MHE = " - MHe: ";
const char* STR_MWI = " - MWi: ";
const char* STR_LEN = "  Length : ";
const char* STR_FSZ = " - F.K_size: ";
const char* STR_FBS = " - F.bias: ";

/*
 * print_data1d_t()
 * Imprime los valores presentes en un vector de datos y su largo
 * Parámetros:
 *            flatten_data_t data => vector de datos a imprimir
 */
void print_data1d_t(const char *head_text, data1d_t data){

    #if EMBEDIA_DEBUG > 0

    PRINT_TXT_LN(head_text);

    PRINT_INT_LN(STR_LEN, data.length);

    #if EMBEDIA_DEBUG > 1

    PRINT_TXT_LN(STR_CONTENT);

    uint16_t i;
    float v;
    for(i=0;i<data.length;i++){

        v = data.data[i];
        PRINT_FL("", v);

    }
    PRINT_TXT_LN("");

    #endif // EMBEDIA_DEBUG 1

    PRINT_TXT_LN(SEP_LINE);

    #endif // EMBEDIA_DEBUG 0

}

void print_data2d_t(const char *head_text, data2d_t data){

    #if EMBEDIA_DEBUG > 0

    PRINT_TXT_LN(head_text);

    PRINT_INT(STR_MHE, data.height);
    PRINT_INT_LN(STR_MWI,data.width);

    #if EMBEDIA_DEBUG > 1

    PRINT_TXT_LN(STR_CONTENT);

    uint16_t h,w;
    for(h=0;h<data.height;h++){
        for(w=0;w<data.width;w++){
            PRINT_FL("", data.data[data.width+h*data.width+w] );
        }
        PRINT_TXT_LN("");
    }
    #endif // EMBEDIA_DEBUG 1

    PRINT_TXT_LN(SEP_LINE);

    #endif // EMBEDIA_DEBUG 0


}

void print_data3d_t(const char *head_text, data3d_t data){


    #if EMBEDIA_DEBUG > 0

    PRINT_TXT_LN(head_text);

    PRINT_INT(STR_CHN, data.channels);
    PRINT_INT(STR_MHE, data.height);
    PRINT_INT_LN(STR_MWI,data.width);

    #if EMBEDIA_DEBUG > 1

    PRINT_TXT_LN(STR_CONTENT);

    uint16_t c,h,w;
    for(c=0;c<data.channels;c++){
        for(h=0;h<data.height;h++){
            for(w=0;w<data.width;w++){
                PRINT_FL("", data.data[c*data.height*data.width+h*data.width+w] );
            }
            PRINT_TXT_LN("");
            }
        PRINT_TXT_LN("");
    }
    #endif // EMBEDIA_DEBUG 1

    PRINT_TXT_LN(SEP_LINE);

    #endif // EMBEDIA_DEBUG 0

}




/*
 * print_filter_t()
 * Imprime los valores de los pesos del filter y sus dimensiones
 * Parámetros:
 *                filter_t filter => filter a imprimir
 */
void print_filter_t(const char *head_text, filter_t filter, uint16_t channels, size2d_t kernel_size){

    #if EMBEDIA_DEBUG > 0

    PRINT_TXT_LN(head_text);

    PRINT_INT_LN(STR_CHN, channels);
    PRINT_INT_LN(STR_FSZ, kernel_size.h);
    PRINT_INT_LN(STR_FSZ, kernel_size.w);
    PRINT_FL_LN(STR_FBS, filter.bias);

    #if EMBEDIA_DEBUG > 1

    printf(STR_CONTENT);

    uint16_t c,h,w;
    for(c=0;c<channels;c++){
        for(h=0;h<kernel_size.h;h++){
            for(w=0;w<kernel_size.w;w++){
                PRINT_FL("", DBG_FL(filter.weights[h*kernel_size.w+w]));
            }
            PRINT_TXT_LN("");
        }
        PRINT_TXT_LN("");
    }
    #endif // EMBEDIA_DEBUG 1

    PRINT_TXT_LN(SEP_LINE);

    #endif // EMBEDIA_DEBUG 0
}
