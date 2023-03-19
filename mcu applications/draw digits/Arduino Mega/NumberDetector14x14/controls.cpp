#include "controls.h"



void Button_Draw( MCUFRIEND_kbv tft, button_t bt, uint16_t clr){
    tft.fillRoundRect(bt.r.x, bt.r.y, bt.r.w, bt.r.h, 8, clr);
    tft.drawRoundRect(bt.r.x, bt.r.y, bt.r.w, bt.r.h, 8, CLR_BLACK);
    tft.setTextColor(CLR_BLACK, clr);
    tft.setTextSize(2);
    int16_t  x1, y1;
    uint16_t w, h;
    tft.getTextBounds(bt.caption, bt.r.x, bt.r.y, &x1, &y1, &w, &h);
    tft.setCursor(bt.r.x+(bt.r.w-w)/2,bt.r.y+(bt.r.h-h)/2+1);
    tft.print(bt.caption);
    
}

uint8_t Button_PointIn(button_t bt, TSPoint pt){
   return bt.r.x < pt.x && pt.x < (bt.r.x+bt.r.w) && bt.r.y < pt.y && pt.y < (bt.r.y+bt.r.h);
}


///////////////////////////////////// screen_image ////////////////////////////////
screen_image_t Screen_Image_Create(uint16_t x, uint16_t y, uint8_t cells_w, uint8_t cells_h, uint8_t cell_sz, uint8_t hdn_cells, fixed * img){
    screen_image_t s = {x,y,cells_w*cell_sz,cells_h*cell_sz, img, cell_sz, cells_w, cells_h,hdn_cells};
    return s;  
}

void Screen_Image_Clear(MCUFRIEND_kbv tft, screen_image_t s, uint16_t bg_clr, uint16_t ln_clr){
    uint16_t i;

    uint16_t hdn_c_sz = s.hdn_cells*s.cell_sz;
    for (i=0; i< s.cells_w*s.cells_h; i++){
        s.image[i]= INT_TO_FIXED(0);
    }
    tft.fillRect(s.r.x+hdn_c_sz, s.r.y+hdn_c_sz, s.r.w-2*hdn_c_sz, s.r.h-2*hdn_c_sz, bg_clr);

    // DrawGrid()

    for (i=s.hdn_cells; i<= s.cells_h-s.hdn_cells; i++){
        //tft.drawFastHLine(CELL_W, BOX_H+i*CELL_H, MAX_SIZE-2*CELL_W, CLR_WHITE); 
        tft.drawFastHLine(s.r.x+hdn_c_sz, s.r.y+i*s.cell_sz, s.r.w-2*hdn_c_sz, ln_clr); 
    }
    for (i=s.hdn_cells; i<= s.cells_w-s.hdn_cells; i++){
        //tft.drawFastVLine(i*CELL_H, BOX_H , MAX_SIZE, CLR_WHITE); 
        tft.drawFastVLine(s.r.x+i*s.cell_sz, s.r.y+hdn_c_sz , s.r.h-2*hdn_c_sz, ln_clr); 
    }

}


void Screen_Image_Draw(MCUFRIEND_kbv tft, screen_image_t s, uint16_t px, uint16_t py, uint8_t sz, uint16_t  bg_clr, uint16_t fg_clr){
   int16_t  r1 = GET_R(bg_clr),g1 = GET_G(bg_clr),b1=GET_B(bg_clr);
   int16_t  r2 = GET_R(fg_clr),g2 = GET_G(fg_clr),b2=GET_B(fg_clr);
   fixed    pxl, dr = INT_TO_FIXED(r2-r1), dg = INT_TO_FIXED(g2-g1), db = INT_TO_FIXED(b2-b1);
   uint16_t color;
   int x,y;

   for(y=0; y < s.cells_h; y++){
       for(x=0; x < s.cells_w; x++){
           pxl=s.image[y*s.cells_w+x];
           color = RBG_TO_COLOR(r1+ FIXED_TO_INT(FIXED_MUL(dr,pxl)), g1+  FIXED_TO_INT(FIXED_MUL(dg,pxl)), b1 + FIXED_TO_INT(FIXED_MUL(db,pxl)) );
           tft.fillRect(px+(x*sz),py+(y*sz), sz, sz, color);
        } 
   }
}


void ScreenImage_SetPixel(MCUFRIEND_kbv tft, screen_image_t s, int16_t x, int16_t y,  uint16_t dot_clr,  uint16_t dot_rad){

    #define PIX(a,b, v) s.image[b*s.cells_w +a] = FIXED_DIV(s.image[b*s.cells_w +a]+v, INT_TO_FIXED(2))
    // screen coord to image coord
    
    uint16_t img_x =  (x-s.r.x) / s.cell_sz;
    uint16_t img_y =  (y-s.r.y) / s.cell_sz;
    s.image[img_y*s.cells_w +img_x] = FL2FX(1.0);

    // Drawing feedback
    int16_t bdr = s.hdn_cells*s.cell_sz;
    x = x - s.r.x - bdr;
    y = y - s.r.y - bdr;
    if ( x < dot_rad)
        x = dot_rad;
    else
        if (x >= (s.r.w-2*bdr-dot_rad))
            x = s.r.w-2*bdr-dot_rad;
    if ( y < dot_rad)
        y = dot_rad;
    else
        if (y >= (s.r.h-2*bdr-dot_rad))
            y = s.r.h-2*bdr-dot_rad;
            
    tft.fillCircle(s.r.x + bdr + x , s.r.y + bdr + y, dot_rad, dot_clr);   
    
}


uint8_t Screen_Image_PointIn(screen_image_t s, TSPoint pt){
   uint16_t hdn = s.hdn_cells*s.cell_sz;
   return s.r.x+hdn <= pt.x && pt.x <= (s.r.x+s.r.w-hdn) && s.r.y+hdn <= pt.y && pt.y <= (s.r.y+s.r.h-hdn);
}

void Text_Append_Char(MCUFRIEND_kbv tft, text_t *t, char c, uint16_t bg_clr, uint16_t fg_clr){
  tft.drawChar(t->r.x+2+t->cursor_pos,t->r.y+4, c, bg_clr, fg_clr, t->text_sz);
  t->cursor_pos+=t->cursor_inc;     
}

void Text_Delete_char(MCUFRIEND_kbv tft, text_t *t, uint16_t bg_clr){
          if (t->cursor_pos > t->cursor_inc){
              t->cursor_pos-= t->cursor_inc;
              tft.fillRect(t->r.x+2+t->cursor_pos,t->r.y+2,t->cursor_inc, t->r.h-2, bg_clr);
          }
}

void Text_Clear(MCUFRIEND_kbv tft, text_t *t, uint16_t bg_clr){
  t->cursor_pos = 2;
  tft.fillRect(t->r.x, t->r.y, t->r.w, t->r.h, bg_clr);
}
