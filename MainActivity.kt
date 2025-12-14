package com.example.plantdiseasedetector

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

data class Disease(val name: String, val cause: String, val cure: String)

class MainActivity : AppCompatActivity() {

    private lateinit var selectBtn: Button
    private lateinit var captureBtn: Button
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var tfliteInterpreter: Interpreter
    private lateinit var diseaseList: ArrayList<Disease>

    private val modelFilename = "plantdiseasemodel.tflite"
    private val labelFilename = "label.txt"

    private var inputImageSize = 224
    private var inputDataType: DataType = DataType.FLOAT32

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.btnSelect)
        captureBtn = findViewById(R.id.btnCapture)
        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.txtResult)

        // Init Model
        try {
            val modelFile = FileUtil.loadMappedFile(this, modelFilename)
            tfliteInterpreter = Interpreter(modelFile)

            val inputTensor = tfliteInterpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            inputImageSize = inputShape[1]
            inputDataType = inputTensor.dataType()

            val jsonString = assets.open(labelFilename).bufferedReader().use { it.readText() }
            val jsonArray = JSONArray(jsonString)
            diseaseList = ArrayList<Disease>()

            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                diseaseList.add(Disease(
                    obj.getString("name"),
                    obj.getString("cause"),
                    obj.getString("cure")
                ))
            }
        } catch (e: Exception) {
            resultText.text = "Error init: ${e.message}"
        }

        // Gallery Button
        selectBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            galleryLauncher.launch(intent)
        }

        // Camera Button
        captureBtn.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                cameraLauncher.launch(null)
            } else {
                requestCameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
            }
        }
    }

    // --- LAUNCHERS ---

    // 1. Gallery Launcher
    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri: Uri? = result.data?.data
            if (uri != null) {
                var bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                // FIX 1: Ensure ARGB_8888 for Gallery images
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                imageView.setImageBitmap(bitmap)
                runInference(bitmap)
            }
        }
    }

    // 2. Camera Launcher
    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            // FIX 2: THIS FIXES THE CRASH "Only supports loading ARGB_8888"
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            imageView.setImageBitmap(argbBitmap)
            runInference(argbBitmap)
        }
    }

    // 3. Permission Launcher
    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            cameraLauncher.launch(null)
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    // --- INFERENCE ---
    private fun runInference(bitmap: Bitmap) {
        try {
            val imageProcessorBuilder = ImageProcessor.Builder()
                .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))

            if (inputDataType != DataType.FLOAT32) {
                imageProcessorBuilder.add(CastOp(inputDataType))
            }

            val imageProcessor = imageProcessorBuilder.build()

            var tensorImage = TensorImage(inputDataType)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)

            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, diseaseList.size), DataType.FLOAT32)
            tfliteInterpreter.run(tensorImage.buffer, outputBuffer.buffer.rewind())

            val probabilities = outputBuffer.floatArray
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

            if (maxIndex != -1) {
                val disease = diseaseList[maxIndex]
                val confidence = probabilities[maxIndex] * 100

                // FIX 3: SMART LOGIC FOR RANDOM OBJECTS
                // If confidence is low (<40%) OR the model explicitly says "Background"
                if (confidence < 40) {
                    resultText.text = "Not sure.\nPlease take a clearer picture of a leaf."
                }
                else if (disease.name.contains("Background", ignoreCase = true)) {
                    resultText.text = "No plant detected.\nPlease take a photo of a plant leaf."
                }
                else {
                    resultText.text = """
                        Detected: ${disease.name}
                        Confidence: %.1f%%

                        CAUSE:
                        ${disease.cause}

                        CURE:
                        ${disease.cure}
                    """.trimIndent().format(confidence)
                }

                Log.d("PlantApp", "Detected: ${disease.name} ($confidence%)")
            } else {
                resultText.text = "Could not identify the image."
            }
        } catch (e: Exception) {
            resultText.text = "Error: ${e.message}"
            Log.e("PlantApp", "Error", e)
        }
    }
}