#ifndef CONTROLS_H
#define CONTROLS_H

#include <stdint.h>
#include <MCUFRIEND_kbv.h>
#include <TouchScreen.h>
#include "fixed.h"


#define RBG_TO_COLOR(r,g,b) ( (((uint16_t)(r)&0xF8)<<8 | ((uint16_t)(g) & 0xFC)<<3 | ((uint16_t)(b)>>3)) )

// Assign human-readable names to some common 16-bit color values:
#define CLR_BLACK   0x0000
#define CLR_BLUE    0x001F
#define CLR_RED     0xF800
#define CLR_GREEN   0x07E0
#define CLR_CYAN    0x07FF
#define CLR_MAGENTA 0xF81F
#define CLR_YELLOW  0xFFE0
#define CLR_WHITE   0xFFFF
#define CLR_MIDWHITE 0b0100001000001000
#define GET_R(RGB) ((RGB) >> 11)
#define GET_G(RGB) (((RGB) >> 5) & 0x003F)
#define GET_B(RGB) ((RGB) & 0x001F)



typedef struct{
    uint16_t x;
    uint16_t y;
    uint8_t w;
    uint8_t h;
} rect_t;

typedef struct{
    rect_t r;
    char caption[8]; 
} button_t;

typedef struct{
    rect_t r;
    fixed *image;
    uint8_t cell_sz;
    uint8_t cells_w;
    uint8_t cells_h; 
    uint8_t hdn_cells;    
} screen_image_t;

typedef struct{
    rect_t r;
    uint8_t text_sz;
    uint16_t cursor_pos;
    uint8_t cursor_inc;
} text_t;


#define Button_Create(x,y,w,h,lbl) {x, y, w, h, lbl}
void Button_Draw( MCUFRIEND_kbv tft, button_t bt, uint16_t clr);
uint8_t Button_PointIn(button_t bt, TSPoint pt);

screen_image_t Screen_Image_Create(uint16_t x, uint16_t y, uint8_t cells_w, uint8_t cells_h, uint8_t cell_sz, uint8_t hdn_cells, fixed * img);

void Screen_Image_Clear( MCUFRIEND_kbv tft, screen_image_t s, uint16_t bg_clr, uint16_t ln_clr);

void Screen_Image_Draw(MCUFRIEND_kbv tft, screen_image_t s, uint16_t px, uint16_t py, uint8_t sz, uint16_t bg_clr, uint16_t fg_clr);

void ScreenImage_SetPixel(MCUFRIEND_kbv tft, screen_image_t s, int16_t x, int16_t y,  uint16_t dot_clr,  uint16_t dot_rad);

uint8_t Screen_Image_PointIn(screen_image_t s, TSPoint pt);

#define Text_Create(x,y,w,h,txt_sz,txt_w) {x, y, w, h, txt_sz, 2, txt_w*txt_sz}
void Text_Append_Char(MCUFRIEND_kbv tft, text_t *t, char c, uint16_t bg_clr, uint16_t fg_clr);
void Text_Delete_char(MCUFRIEND_kbv tft, text_t *t, uint16_t bg_clr);
void Text_Clear(MCUFRIEND_kbv tft, text_t *t, uint16_t bg_clr);

#endif
