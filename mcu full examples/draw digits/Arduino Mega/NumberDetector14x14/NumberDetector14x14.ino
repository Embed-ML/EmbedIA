/*
 * This example runs on an Arduino Mega and requires a display to be able 
 * to recognize handwritten digits. The convolutional network was trained
 * with the MNIST dataset. For space reasons the size of the original
 * dataset was reduced and a limited convolutional network model was built
 * to run in 8Kib of RAM.
 *
 */
 
#include "mnist_digits_model.h"

#include <Adafruit_GFX.h>    // Core graphics library
#include <Adafruit_TFTLCD.h> // Hardware-specific library
#include <TouchScreen.h>
#include <MCUFRIEND_kbv.h> // Touchscreen Hardware-specific library
#include "controls.h"


#if defined(__SAM3X8E__)
    #undef __FlashStringHelper::F(string_literal)
    #define F(string_literal) string_literal
#endif


#define YP A1  // must be an analog pin, use "An" notation!
#define XM A2  // must be an analog pin, use "An" notation!
#define YM 7   // can be a digital pin
#define XP 6   // can be a digital pin

#define TS_MINX 140
#define TS_MINY 140
#define TS_MAXX 900 
#define TS_MAXY 950

// For better pressure precision, we need to know the resistance
// between X+ and X- Use any multimeter to read it
// For the one we're using, its 300 ohms across the X plate
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

#define LCD_CS A3 // Chip Select goes to Analog 3
#define LCD_CD A2 // Command/Data goes to Analog 2
#define LCD_WR A1 // LCD Write goes to Analog 1
#define LCD_RD A0 // LCD Read goes to Analog 0

#define LCD_RESET A4 // Can alternately just connect to Arduino's reset pin

MCUFRIEND_kbv tft;

#define MAX_SIZE 240
#define BOX_SIZE (MAX_SIZE/9)
#define PEN_RADIUS 15

#define BOX_H 38
#define BOX_W 38

#define IMG_W INPUT_WIDTH
#define IMG_H INPUT_HEIGHT

#define MINPRESSURE 10
#define MAXPRESSURE 1000
#define TS_RELEASE_TIME 750

#define PXL_SZ 3
  
#define TEXT_SIZE 4
#define TEXT_SPACE 6
#define BG_CLR RBG_TO_COLOR(21, 67, 128)
#define BT1_CLR RBG_TO_COLOR(64,255,64)
#define BT2_CLR RBG_TO_COLOR(255, 92, 92)
#define BT3_CLR RBG_TO_COLOR(255,255,64)
#define BNT_Y (320-BOX_H-2)
#define BNT_X (5+2*PXL_SZ*IMG_W)


fixed image[INPUT_CHANNELS*INPUT_WIDTH*INPUT_HEIGHT];
data3d_t input = {INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT, image};

button_t bt_image = Button_Create(BNT_X,BNT_Y, BOX_W, BOX_H-4, "**");
button_t bt_clear = Button_Create(BNT_X +BOX_H+2,BNT_Y, BOX_W, BOX_H-4, "<<");
button_t bt_back  = Button_Create(BNT_X +2*(BOX_H+2),BNT_Y, BOX_W, BOX_H-4, "<-");

screen_image_t screen_img;

text_t txt_feed = Text_Create(0, 2, MAX_SIZE,BOX_H, TEXT_SIZE, TEXT_SPACE);

void setup(void) {
  Serial.begin(115200);
  Serial.println(F("MNIST Handwritten Recognition Example on Mega 2560"));

  tft.reset();
  
  uint16_t identifier = tft.readID();
 // identifier == 0x9341
  if(identifier == 0x9325) {
    Serial.println(F("Found ILI9325 LCD driver"));
  } else if(identifier == 0x9328) {
    Serial.println(F("Found ILI9328 LCD driver"));
  } else if(identifier == 0x7575) {
    Serial.println(F("Found HX8347G LCD driver"));
  } else if(identifier == 0x9341) {
    Serial.println(F("Found ILI9341 LCD driver"));
  } else if(identifier == 0x8357) {
    Serial.println(F("Found HX8357D LCD driver"));
  } else {
    Serial.print(F("Unknown LCD driver chip: "));
    Serial.println(identifier, HEX);
    return;
  }
    

  tft.begin(identifier);
  tft.setRotation(0);

  pinMode(LED_BUILTIN, OUTPUT);

  tft.fillRect(0, 0, tft.width(), tft.height(), BG_CLR);
  tft.drawFastHLine(0, 2+BOX_H, MAX_SIZE, CLR_WHITE); 

  uint16_t cells_sz = (MAX_SIZE-1) / min(IMG_W,IMG_H);
  uint16_t x = (MAX_SIZE-cells_sz*IMG_W)/2, y = (MAX_SIZE-cells_sz*IMG_H)/2;

  screen_img = Screen_Image_Create(x, y+BOX_H, IMG_W, IMG_H, cells_sz, 1, image);

  Screen_Image_Clear(tft, screen_img, CLR_BLACK, CLR_MIDWHITE);
  
  Button_Draw(tft, bt_image, BT1_CLR);
  Button_Draw(tft, bt_clear, BT2_CLR);
  Button_Draw(tft, bt_back,  BT3_CLR);

  model_init();
   
}


long ts_time;

bool touching = false;
void loop(){

  TSPoint p = ts.getPoint();

  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);
  
  // check touchpad pressure
  if (p.z > MINPRESSURE && p.z < MAXPRESSURE) {

      // scale from 0->1023 to tft.width
      p.x = map(p.x, TS_MINX, TS_MAXX, tft.width(),0);
      p.y = map(p.y, TS_MINY, TS_MAXY, tft.height(),0);
     
      // check if point is in control
      if (Button_PointIn(bt_image, p)){
          Screen_Image_Clear(tft, screen_img, CLR_BLACK, CLR_MIDWHITE);
      }
      else if (Button_PointIn(bt_back, p)){
          Text_Delete_char(tft, &txt_feed, BG_CLR);
          delay(100);
      }
      else if (Button_PointIn(bt_clear, p)){
          Text_Clear(tft, &txt_feed, BG_CLR);
      }
      else {
          if ( Screen_Image_PointIn(screen_img, p) ){
              ts_time = millis();
              if (!touching){        
                  touching = true;
                  SetRecordingLed(CLR_RED);
              }
              ScreenImage_SetPixel( tft, screen_img, p.x, p.y,  CLR_WHITE,  PEN_RADIUS);
          }         
      }
  }
  else{
      if (touching){
          long ts_endtime= millis();
          // check ellapsed time without touching
          if ( ts_endtime-ts_time > TS_RELEASE_TIME){
              touching = false;
              SetRecordingLed(CLR_GREEN);
              
              BlurImage();
              
              data1d_t resultados; 
              uint8_t prediccion = model_predict_class(input,&resultados);
              char c;
              if (prediccion < 10)
                  c = char(prediccion+48);
              else
                  c = char(prediccion-10+65);
              Serial.println(c);
              Text_Append_Char(tft, &txt_feed, c, CLR_WHITE, BG_CLR);            
          }
      }
  }
}

void SetRecordingLed( uint16_t color){
   tft.fillRect(tft.width()-2*BOX_H/3-1, tft.height()-2*BOX_H/3-1-6, BOX_H/2, BOX_H/2, color);
}

void BlurImage(){
  uint16_t x, y;
  fixed pxl;

  Screen_Image_Draw(tft, screen_img, 1,  tft.height()-PXL_SZ*IMG_H-2, PXL_SZ, CLR_BLACK, CLR_WHITE);

  for(y=0; y < IMG_H-2; y++){
    for(x=0; x < IMG_W-2; x++){
        pxl =  FIXED_MUL(image[y*IMG_W+x]+image[y*IMG_W+x+1]+image[(y+1)*IMG_W+x]+image[(y+1)*IMG_W+x+1], FL2FX(1.0/4));
        if (pxl > image[y*IMG_W+x])
            image[y*IMG_W+x] = pxl;
    }
  }

  Screen_Image_Draw(tft, screen_img, 1+1+PXL_SZ*IMG_W, tft.height()-PXL_SZ*IMG_H-2, PXL_SZ, CLR_BLACK, CLR_YELLOW);

}
