#include <Wire.h>
#include <VL53L1X.h>
#include <Servo.h>

String (InBytes);

Servo servo;

VL53L1X sensor;

void setup()
{
  servo.attach(9);
  Serial.begin(115200); // clean and single baud setup
  Wire.begin();
  Wire.setClock(400000);

  sensor.setTimeout(500);
  if (!sensor.init()) {
    Serial.println("Failed to detect and initialize sensor!");
    while (1);
  }

  sensor.setDistanceMode(VL53L1X::Medium);
  sensor.setROICenter(199);
  sensor.setROISize(4, 4);
  sensor.setMeasurementTimingBudget(10000);  // 10 ms
  sensor.startContinuous(10);
  Serial.println("READY");
  Serial.println("new program");
}

void loop() {
  Serial.print(millis());
  Serial.print(',');
  Serial.println(sensor.read());
}
