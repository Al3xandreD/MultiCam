#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>
 
int L_WEIGHT_BUTTON_PIN = 14;
int H_WEIGHT_BUTTON_PIN = 2;
int wifi_choice = 0;
const char* WIFI_SSID[] = {"iot", "dlinkMALEK", "HUAWEI nova 3i", "Redmi Note 8 Pro"};
const char* WIFI_PASS[] = {"enstaL@b", "felmw98786", "12345678","12345678"};

WebServer server(80);


/*
FRAMESIZE_UXGA (1600 x 1200)
FRAMESIZE_QVGA (320 x 240) 
FRAMESIZE_CIF (352 x 288)  
FRAMESIZE_VGA (640 x 480) 
FRAMESIZE_SVGA (800 x 600) 
FRAMESIZE_XGA (1024 x 768) 
FRAMESIZE_SXGA (1280 x 1024)
*/

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
 
 
void  setup(){
  Serial.begin(115200);
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

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
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











