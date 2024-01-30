#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

//WIFI connection status LED
#define BUILT_IN_WHITE_LED_PIN 4 
#define PWMChannel 7 // Use PWM channel 7
#define LEDC_TIMER_BIT_RES  12 // Use 12 bit precision for LEDC timer
#define LEDC_BASE_FREQ     5000 // Use 5000 Hz as a LEDC base frequency
#define LED_INTENSITY 2 // Set duty cycle --> Set led Intensity 230.

//WIFI choice buttons (2 buttons --> 4 combinaisons --> 4 Wifi choices)
#define L_WEIGHT_BUTTON_PIN  14
#define H_WEIGHT_BUTTON_PIN  2

int wifi_choice = 0; //Range : 0 - 3
const char* WIFI_SSID[] = {"iot", "HUAWEI nova 3i", "RGX Phone", "dlinkMALEK"};
const char* WIFI_PASS[] = {"enstaL@b", "12345678","12345678", "felmw98786"};


WebServer server(80); //80 --> http



static auto loRes = esp32cam::Resolution::find(320, 240);
static auto midRes = esp32cam::Resolution::find(350, 530);
static auto hiRes = esp32cam::Resolution::find(1600, 1200);

void serveJpg()
{
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));
 
  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}
 
void handleJpgLo()
{
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  serveJpg();
}
 
void handleJpgHi()
{
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}
 
void handleJpgMid()
{
  if (!esp32cam::Camera.changeResolution(midRes)) {
    Serial.println("SET-MID-RES FAIL");
  }
  serveJpg();
}
 
 /*
FRAMESIZE_UXGA (1600 x 1200)
FRAMESIZE_QVGA (320 x 240) 
FRAMESIZE_CIF (352 x 288)  
FRAMESIZE_VGA (640 x 480) 
FRAMESIZE_SVGA (800 x 600) 
FRAMESIZE_XGA (1024 x 768) 
FRAMESIZE_SXGA (1280 x 1024)
*/
void  setup(){

  //Serial comm setup
  Serial.begin(115200);

  //Built in Led Setup
  ledcAttachPin(BUILT_IN_WHITE_LED_PIN, PWMChannel);
  ledcSetup(PWMChannel, LEDC_BASE_FREQ, LEDC_TIMER_BIT_RES); 
  ledcWrite(PWMChannel, 0);   //Turn off (0% Duty Cycle)

  //Camera Setup
  Serial.println();
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(80);  //Quality : 0 to 100
 
    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  pinMode(L_WEIGHT_BUTTON_PIN, INPUT_PULLUP);
  pinMode(H_WEIGHT_BUTTON_PIN, INPUT_PULLUP);

  wifi_choice = !digitalRead(L_WEIGHT_BUTTON_PIN) + 2*!digitalRead(H_WEIGHT_BUTTON_PIN);
  WiFi.begin(WIFI_SSID[wifi_choice], WIFI_PASS[wifi_choice]);
  Serial.println("Trying to connect to Wifi " + String(wifi_choice + 1) + " :");
  Serial.print(WIFI_SSID[wifi_choice]);

  //Print dots and blink the white led until successful connection
  bool toogleWhiteLed = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    toogleWhiteLed = !toogleWhiteLed;
    ledcWrite(PWMChannel, toogleWhiteLed ? 0 : LED_INTENSITY); 
  }
  
  ledcWrite(PWMChannel, 0);   //  Turn off white led to indicate successful connected status
  Serial.println(" ");
  Serial.println("Wifi Connected ! ");
  Serial.print("http://");
  Serial.println(WiFi.localIP());

  Serial.println("  /cam-lo.jpg");
  Serial.println("  /cam-hi.jpg");
  Serial.println("  /cam-mid.jpg");
 
  server.on("/cam-lo.jpg", handleJpgLo);
  server.on("/cam-hi.jpg", handleJpgHi);
  server.on("/cam-mid.jpg", handleJpgMid);
 
  server.begin();
}
 
void loop()
{
  server.handleClient();
}











