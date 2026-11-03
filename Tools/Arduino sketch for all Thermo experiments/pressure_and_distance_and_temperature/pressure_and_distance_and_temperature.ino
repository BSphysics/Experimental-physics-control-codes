#include "Adafruit_VL53L1X.h"
#include <Wire.h>
#include <SparkFun_MicroPressure.h>
#include <max6675.h>


#define IRQ_PIN 2
#define XSHUT_PIN 3

Adafruit_VL53L1X vl53 = Adafruit_VL53L1X(XSHUT_PIN, IRQ_PIN);

//SparkFun_MicroPressure mpr(EOC_PIN, RST_PIN, MIN_PSI, MAX_PSI);
SparkFun_MicroPressure mpr; // Use default values with reset and EOC pins unused

// MAX6675 temperature sensor config
int thermoCS = 10;    // chip select pin
int thermoSCK = 13;  //clock
int thermoSO = 12;   //MISO

int thermoCS2 = 9;    // chip select pin

MAX6675 thermocouple(thermoSCK, thermoCS, thermoSO);

//MAX6675 thermocouple2(thermoSCK, thermoCS2, thermoSO);


void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Wire.begin();

  if (! vl53.begin(0x29, &Wire)) {
    Serial.print(F("Error on init of VL sensor: "));
    Serial.println(vl53.vl_status);
    while (1)       delay(10);
  }

  if (! vl53.startRanging()) {
    Serial.print(F("Couldn't start ranging: "));
    Serial.println(vl53.vl_status);
    while (1)       delay(10);
  }


  // Valid timing budgets: 15, 20, 33, 50, 100, 200 and 500ms!
  vl53.setTimingBudget(500);

  vl53.VL53L1X_SetDistanceThreshold(1, 500, 3, 1);

  delay(500);


  if (!mpr.begin())
  {
    Serial.println("Cannot connect to MicroPressure sensor.");
    while (1);
  }
}

void loop() {
  int16_t distance;

  if (vl53.dataReady()) {
    // new measurement for the taking!
    distance = vl53.distance();
    if (distance == -1) {
      // something went wrong!
      Serial.print(F("Couldn't get distance: "));
      Serial.println(vl53.vl_status);
      return;
    }
    //Serial.print(F("Distance: "));
    Serial.println(distance);
    //Serial.println(" mm");

    Serial.println(mpr.readPressure(PA), 1);
    //Serial.println(" Pa");
    delay(1);

    float temperature = thermocouple.readCelsius() + 273.15;
    //float temperature2 = thermocouple2.readCelsius();

    //float tempDiff = temperature - temperature2;

    //Serial.println(tempDiff);
    Serial.println(temperature);
    //Serial.println(temperature2);

    // data is read out, time for another reading!
    vl53.clearInterrupt();
  }
}
