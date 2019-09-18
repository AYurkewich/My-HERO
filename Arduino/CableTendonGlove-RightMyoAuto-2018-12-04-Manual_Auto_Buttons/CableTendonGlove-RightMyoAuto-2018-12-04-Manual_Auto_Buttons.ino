/*
   Copyright (c) 2016 Intel Corporation.  All rights reserved.
   See the bottom of this file for the license terms.
*/

/*
   This sketch example demonstrates how the BMI160 on the
   Intel(R) Curie(TM) module can be used to read accelerometer data
*/

#include "CurieIMU.h"
#include <Servo.h>
#include <CurieBLE.h>

Servo myservoflex;  // create servo object to control a servo
Servo myservoextend;  // create servo object to control a servo

BLEService motorService("ebbb19a6-c943-44f9-aee0-e180300007f0"); //custom 120-bit UUID custom service

//motor characteristic
BLEIntCharacteristic motorReading("00211321-dc03-4f55-82b6-14a630bd8e2d",
                                  BLERead | BLENotify | BLEWrite);

//motor properties
BLEIntCharacteristic motorExtendChar("809ba7a9-13ad-4446-a005-bdc12ca93c76", BLERead | BLENotify | BLEWrite);
BLEIntCharacteristic motorContractChar("2fad8a3f-e1a1-47cd-982b-24e13fbe9342", BLERead | BLENotify | BLEWrite);

// twelve servo objects can be created on most boards

float timeMillis = 0;
float timeMillisMotionReset = 0;
float timeMillisGyroReset = 0;
float ax, ay, az, atot, gx, gy, gz, gtot, gx_bias, gy_bias, gz_bias, gtot_filt;
int motionDetect = 0;
int switchPosition = 0; // fingers extended
int autoButton = 0;
int autoButtonOld = 0;
int extendButton = 0;
int extendButtonOld = 0;
int extendButtonChange = 0;
int delayMode = 0;
int timeMillisDelayMode = 0;
int extendMotor = 55; //50 = fully squeeze bottle; 80 = partially squeeze bottle
int retractMotor = 150; //150 = fully extend fingers; 120 = partially extend fingers
int manualExtend = 0;
int manualContract = 1;
int manualRelax = -1;

void setup() {
  myservoflex.attach(A1);  // attaches the servo
  myservoextend.attach(A3);  // attaches the servo
  myservoflex.write(extendMotor);
  myservoextend.write(extendMotor);
  pinMode(2, INPUT);
  pinMode(3, OUTPUT);
  digitalWrite(3, LOW);
  pinMode(4, INPUT);
  pinMode(6, INPUT);
  pinMode(7, OUTPUT);
  digitalWrite(7, LOW);
  pinMode(8, INPUT);
  //myservoflex.write(40);
  //myservoextend.write(40);
  //Serial.begin(9600); // initialize Serial communication
  //while (!Serial);    // wait for the serial port to open
  timeMillis = millis();
  timeMillisMotionReset = millis();
  digitalWrite(13, LOW);
  CurieIMU.begin();
  // Set the accelerometer range to 2G
  CurieIMU.setAccelerometerRange(2);
  // Set the accelerometer range to 250 degrees/second
  CurieIMU.setGyroRange(250);

  BLE.begin();

  BLE.setLocalName("LegoHERO");

  BLE.setAdvertisedService(motorService);  // add the service UUID
  motorService.addCharacteristic(motorReading);
  motorService.addCharacteristic(motorExtendChar);
  motorService.addCharacteristic(motorContractChar);
  BLE.addService(motorService);   // Add the BLE Battery service

  motorReading.setValue(manualRelax);
  motorExtendChar.setValue(extendMotor);
  motorContractChar.setValue(retractMotor);

  extendMotor = motorExtendChar.value();
  retractMotor = motorContractChar.value();

  BLE.advertise();

  //read buttons
  if (digitalRead(2) == HIGH)
  {
    autoButton = 1;
  }
  else
  {
    autoButton = 0;
  }
  if (digitalRead(6) == HIGH)
  {
    extendButton = 1;
  }
  else
  {
    extendButton = 0;
  }

  // if auto button is pressed down we are in manual mode, start manual mode with motors relaxed
  if (autoButton == 1)
  { myservoflex.write(retractMotor);
    myservoextend.write(retractMotor);
  }

  // if auto button is pressed up we are in auto mode, start auto mode with relaxed position
  else if (autoButton == 0)
  {
    myservoflex.write(retractMotor);
    myservoextend.write(retractMotor);
    delayMode = 1;
    timeMillisDelayMode = millis();
  }
  //Serial.begin(9600);
}

void loop() {
  //on loop reset the motor values
  extendMotor = motorExtendChar.value();
  retractMotor = motorContractChar.value();

  // read accelerometer measurements from device, scaled to the configured range
  CurieIMU.readAccelerometerScaled(ax, ay, az);
  CurieIMU.readGyroScaled(gx, gy, gz);

  //calculate absolute value of gyro
  gtot = sqrt(sq(gx - gx_bias) + sq(gy - gy_bias) + sq(gz - gz_bias));
  //filter gtot so quick noise is not detected as motion
  gtot_filt = gtot * 0.2 + gtot_filt * 0.8;

  //Serial.println(motorExtendChar.value());
  //Serial.println(motorContractChar.value());
  //Serial.print(motorReading.value());
  if (motorReading.value()== 1)
    {myservoflex.write(retractMotor);
    myservoextend.write(extendMotor);}
  else if (motorReading.value()== 0)
    {myservoflex.write(extendMotor);
    myservoextend.write(retractMotor);}
  else
    {myservoflex.write(extendMotor);
    myservoextend.write(extendMotor);}
  //some hope for communication through voice

  //  open();
  //  TestX.write(gx-gx_bias);
  //  TestY.write(gy-gy_bias);
  //  TestZ.write(gz-gz_bias);

//  Serial.print(gx-gx_bias);
//  Serial.print("\t");
//  Serial.print(gy-gy_bias);
//  Serial.print("\t");
//  Serial.println(gz-gz_bias);

//  Serial.print(ax);
//  Serial.print("\t");
//  Serial.print(ay);
//  Serial.print("\t");
//  Serial.println(az);

  //Serial.println("\t");
  //Serial.println(motionDetect);
  //Serial.println(gtot);

  //read time
//  timeMillis = millis();

  //read buttons
  //read buttons
//  if (digitalRead(2) == HIGH)
//  {
//    autoButton = 1;
//  }
//  else
//  {
//    autoButton = 0;
//  }
//  if (digitalRead(6) == HIGH)
//  {
//    extendButton = 1;
//  }
//  else
//  {
//    extendButton = 0;
//  }
//
//  //reset gyro biases when user is not moving, so that when the user is at rest gtot is 0
//  if ((gtot < 5) && (timeMillis > timeMillisGyroReset + 20000)) {
//    gx_bias = gx;
//    gy_bias = gy;
//    gz_bias = gz;
//    timeMillisGyroReset = millis();
//  }
//
//  //delay mode, don't trigger robot to move
//  if ((autoButton == 0) && (delayMode == 1) && (timeMillis < (timeMillisDelayMode + 2000)))
//  {
//    motionDetect = 0;
//  }
//
//  else {
//    delayMode = 0;
//    //detect if auto button has been pressed; autoButton=0 is auto mode (button up), autoButton=1 is manual mode (button down)
//    if (autoButton != autoButtonOld) {
//      // if auto button is pressed down we are in manual mode, start manual mode with motors relaxed
//      if (autoButton == 1)
//      {
//        motorReading.setValue(manualRelax);
//      }
//
//      // if auto button is pressed up we are in auto mode, start auto mode with fingers extended
//      else if (autoButton == 0)
//      { myservoflex.write(retractMotor);
//        myservoextend.write(extendMotor);
//        switchPosition = 0;
//        delayMode = 1;
//        timeMillisDelayMode = millis();
//      }
//
//      autoButtonOld = autoButton;
//      extendButtonOld = extendButton;
//      motionDetect = 0;
//      //delay(7000);
//    }
//
//    //detect if robot is in auto mode
//    if (autoButton == 0) {
//      //detect if angular motion is greater than 15 units. If so, keep resetting the reset counter while the user is moving
//      if (gtot > 100) {
//        motionDetect = 1;
//        timeMillisMotionReset = millis();
//      }
//
//      //after moving, if the user isd stationary for long enough, trigger motor to move
//      if ((motionDetect == 1) && (timeMillis > (timeMillisMotionReset + 800))) {
//        if (switchPosition == 1) {
//          myservoflex.write(retractMotor);
//          myservoextend.write(extendMotor);
//          switchPosition = 0;
//        }
//        else {
//          myservoflex.write(extendMotor);
//          myservoextend.write(retractMotor);
//          switchPosition = 1;
//        }
//        delayMode = 1;
//        timeMillisDelayMode = millis();
//      }
//    }
//
//    //detect if extend button is pressed while robot is in manual mode
//    if ((autoButton == 1) && (extendButton != extendButtonOld)) {
//      extendButtonOld = extendButton;
//      extendButtonChange = 1;
//    }
//
//    //if extend button is pressed while robot is in manual mode then trigger finger extension if extend button is pressed up or flexion if extend button pressed down
//    else if ((autoButton == 1) && (extendButtonChange == 1) && (extendButton == 1)) {
//      /*
//        myservoflex.write(extendMotor);
//        myservoextend.write(retractMotor);
//      */
//      motorReading.setValue(manualExtend);
//      extendButtonChange = 0;
//    }
//
//    else if ((autoButton == 1) && (extendButtonChange == 1) && (extendButton == 0)) {
//      /*
//        myservoflex.write(retractMotor);
//        myservoextend.write(extendMotor);
//      */
//      motorReading.setValue(manualContract);
//      extendButtonChange = 0;
//
//    }
//
//
//
//    if (autoButton == 1) {
//      //update motors based off of the new characteristic values presented
//      if ( motorReading.value() == manualContract) {
//        myservoflex.write(retractMotor);
//        myservoextend.write(extendMotor);
//      }
//
//      if ( motorReading.value() == manualExtend) {
//        myservoflex.write(extendMotor);
//        myservoextend.write(retractMotor);
//      }
//
//      if ( motorReading.value() == manualRelax) {
//        myservoflex.write(extendMotor);
//        myservoextend.write(extendMotor );
//
//      }
//    }
//
//  }//else for delay mode
}//end of void loop

/*
   Copyright (c) 2016 Intel Corporation.  All rights reserved.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/
