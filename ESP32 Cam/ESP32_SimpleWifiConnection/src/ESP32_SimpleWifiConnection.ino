#include <WiFi.h>


const int WIFI_CHOICE = 0;
const char* WIFI_SSID[] = {"HUAWEI nova 3i", "Remi Note 8 Pro"};
const char* WIFI_PASS[] = {"12345678","12abc345"};
const char* ssid = WIFI_SSID[WIFI_CHOICE];
const char* password = WIFI_PASS[WIFI_CHOICE];

void setup(){
    Serial.begin(115200);
    delay(1000);

    WiFi.mode(WIFI_STA); //Optional
    WiFi.begin(ssid, password);
    Serial.println("\nConnecting");

    while(WiFi.status() != WL_CONNECTED){
        Serial.print(".");
        delay(100);
    }

    Serial.println("\nConnected to the WiFi network");
    Serial.print("Local ESP32 IP: ");
    Serial.println(WiFi.localIP());
}

void loop(){}