# Meeter reader android SDK
---
Tensorflow lite android SDK for meeter reader. There are 3 machine learning model used here.
| Model      | Description |
| ----------- | ----------- |
| app/src/main/assets/display_detection.tflite      | Model for meeter display detection       |
| app/src/main/assets/reading_detectionV2.tflite   | Model for meeter reading detection        |
| app/src/main/assets/reading_ocrV3.tflite   | Model for meeter reading OCR        |

## Getting Started
This Guide will walk you through the steps needed to start using meeter reader SDK in your app, including running inference on your first image.

>You can either download the AAR file here [meeter_reader_sdk-v1-release.aar](https://github.com/renjithsasidharan/meeter-reader-sdk/blob/main/meeter_reader_sdk-v1-release.aar) or build from source as described below.

### Building the library
Open a terminal and run:
```sh
./gradlew clean assembleRelease
```
Output:
```sh
❯ ./gradlew clean assembleRelease

> Configure project :app
Warning: The 'kotlin-android-extensions' Gradle plugin is deprecated. Please use this migration guide (https://goo.gle/kotlin-android-extensions-deprecation) to start working with View Binding (https://developer.android.com/topic/libraries/view-binding) and the 'kotlin-parcelize' plugin.
Warning: Mapping new ns http://schemas.android.com/repository/android/common/02 to old ns http://schemas.android.com/repository/android/common/01
Warning: Mapping new ns http://schemas.android.com/repository/android/generic/02 to old ns http://schemas.android.com/repository/android/generic/01
Warning: Mapping new ns http://schemas.android.com/sdk/android/repo/addon2/02 to old ns http://schemas.android.com/sdk/android/repo/addon2/01
Warning: Mapping new ns http://schemas.android.com/sdk/android/repo/repository2/02 to old ns http://schemas.android.com/sdk/android/repo/repository2/01
Warning: Mapping new ns http://schemas.android.com/sdk/android/repo/sys-img2/02 to old ns http://schemas.android.com/sdk/android/repo/sys-img2/01
aapt2 W 03-08 20:52:51 83823 906398 LoadedArsc.cpp:657] Unknown chunk type '200'.


> Task :app:kaptGenerateStubsReleaseKotlin
w: The '-jdk-home' option is ignored because '-no-jdk' is specified

> Task :app:compileReleaseKotlin
w: The '-jdk-home' option is ignored because '-no-jdk' is specified

Deprecated Gradle features were used in this build, making it incompatible with Gradle 7.0.
Use '--warning-mode all' to show the individual deprecation warnings.
See https://docs.gradle.org/6.8.3/userguide/command_line_interface.html#sec:command_line_warnings

BUILD SUCCESSFUL in 27s
27 actionable tasks: 24 executed, 3 up-to-date
```

The AAR file will be created at the location:
```sh
❯ app/build/outputs/aar/meeter_reader_sdk-v{versionCode}-release.aar
```
The `versionCode` is defined in, [build.gradle](https://github.com/renjithsasidharan/meeter-reader-sdk/blob/main/app/build.gradle)

## Usage
### Update Gradle configuration
1. Copy AAR file `meeter_reader_sdk-v{versionCode}-release.aar` to `app/libs`
2. Edit your `build.gradle` file. You must add the following line to the `dependencies` section:
    ```java
    dependencies {
      // Import these for meeter reader sdk
      implementation fileTree(dir: 'libs', include: ['*.aar'])
      implementation 'org.tensorflow:tensorflow-lite:2.8.0'
      implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.8.0'
      implementation 'org.tensorflow:tensorflow-lite-support:0.3.1'
      implementation 'com.quickbirdstudios:opencv:4.3.0'
    }
    ```
  3. Edit your `build.gradle` file. You must add the following line to `plugins` sections:
      ```java
      plugins {
          // Add these lines
          id 'kotlin-android'
          id 'kotlin-android-extensions'
          id 'kotlin-kapt'
      }
      ```
  4. Edit your `build.gradle` file. You must add the following line to `android` sections:
      ```java
      aaptOptions {
        noCompress "tflite"
      }
      ```
  ### Initialize SDK
  Meeter reader SDK needs to be initialized like `OCRModelExecutor(context)`. You should only do this 1 time, so placing the initialization in your Application is a good idea. An example for this would be:

```kotlin
private suspend fun createModelExecutor() {
    mutex.withLock {
        if (ocrModel != null) {
            ocrModel!!.close()
            ocrModel = null
        }
        try {
            ocrModel = OCRModelExecutor(this)
        } catch (e: Exception) {
            Log.e(TAG, "Fail to create OCRModelExecutor: ${e.message}")
        }
    }
}
```
### Run inference on a bitmap image
To run inference on an image, you need to do call `OCRModelExecutor.execute(Bitmap)`. 

`OCRModelExecutor.execute(Bitmap)` takes a `Bitmap` image as input where:
1.  Image `width` should be equal to its `height`
2.  Image `width` and `height` should be at least `320` pixels.
3.  Image `width` and `height` should be a multiple of `32`. For ex: `320`, `512` etc.
>  If you pass a Bitmap of size greater than `320` (for ex: `512`), it will be resized to `320` internally. So it always better to pass an image of size `320`.

An example would be:
```kotlin
private val viewModelJob = Job()
private val viewModelScope = CoroutineScope(viewModelJob)

// the execution of the model has to be on the same thread where the interpreter
// was created
fun onApplyModel(
    contentImage: Bitmap,
    ocrModel: OCRModelExecutor?,
    inferenceThread: ExecutorCoroutineDispatcher
) {
    viewModelScope.launch(inferenceThread) {
        try {
            val result = ocrModel?.execute(contentImage)
        } catch (e: Exception) {
            Log.e(TAG, "Fail to execute OCRModelExecutor: ${e.message}")
        }
    }
}
```

The result of `OCRModelExecutor.execute()` is an instance of `ModelExecutionResult`. object which has the following properties:
| Type           | Field            | Description                               |
|----------------|------------------|-------------------------------------------|
| ``val Bitmap`` | ``bitmapResult`` | The annotated image.                      |
| ``val String`` | ``executionLog`` | Contains error messages or info messages. |
| ``val String`` | ``reading``      | The meeter reading.                       |

