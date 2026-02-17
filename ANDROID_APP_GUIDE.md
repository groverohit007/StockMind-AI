# StockMind-AI: Android App Development Guide

A step-by-step guide to convert the StockMind-AI stock prediction tool into a native Android application.

---

## Architecture Overview

```
Android App (Kotlin/Jetpack Compose)
        |
        v
  REST API Layer (Python FastAPI / Flask)
        |
        v
  ML Prediction Engine (existing logic.py)
        |
        v
  Data Sources (yfinance, Alpha Vantage)
```

The recommended architecture separates the app into two parts:
1. **Backend API** - Hosts the ML models and prediction logic (Python)
2. **Android Client** - Native app that consumes the API (Kotlin)

Running ML models directly on-device is impractical here because the ensemble uses XGBoost, LightGBM, CatBoost, and scikit-learn, which are not available natively on Android.

---

## Step 1: Build the Backend API

Convert the existing Streamlit app into a REST API that the Android app will call.

### 1.1 Install FastAPI dependencies

```bash
pip install fastapi uvicorn pydantic
```

### 1.2 Create `api_server.py`

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import logic
import database as db

app = FastAPI(title="StockMind-AI API", version="1.0")
security = HTTPBearer()

# ---------- Models ----------

class PredictionRequest(BaseModel):
    ticker: str
    timeframes: List[str] = ["1d", "1wk"]

class PredictionResponse(BaseModel):
    ticker: str
    timeframe: str
    signal: str          # BUY / SELL / HOLD
    confidence: float    # 0.0 - 1.0
    model_accuracy: float
    current_price: float
    target_price: Optional[float]

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    token: str
    tier: str  # free / premium

# ---------- Auth ----------

@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = db.authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = db.create_session_token(user["id"])
    return TokenResponse(token=token, tier=user["subscription_tier"])

# ---------- Predictions ----------

@app.post("/api/v1/predict", response_model=List[PredictionResponse])
def predict(req: PredictionRequest, creds: HTTPAuthorizationCredentials = Depends(security)):
    user = db.verify_token(creds.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    results = []
    for tf in req.timeframes:
        data = logic.get_data(req.ticker, period="2y", interval=tf)
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {req.ticker}")

        model_result = logic.train_ultimate_model(data, req.ticker, tf)
        prediction = logic.make_prediction(model_result, data)

        results.append(PredictionResponse(
            ticker=req.ticker,
            timeframe=tf,
            signal=prediction["signal"],
            confidence=prediction["confidence"],
            model_accuracy=model_result["accuracy"],
            current_price=float(data["Close"].iloc[-1]),
            target_price=prediction.get("target_price"),
        ))
    return results

@app.get("/api/v1/watchlist/scan")
def scan_watchlist(creds: HTTPAuthorizationCredentials = Depends(security)):
    user = db.verify_token(creds.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    # Reuse existing scanner logic
    # Return list of signals for watchlist stocks
    pass

# ---------- Run ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 1.3 Deploy the API

Options for hosting the Python API:
- **Railway.app** - Easy Python deployment, free tier available
- **Render.com** - Free tier with auto-deploy from GitHub
- **AWS EC2 / GCP Compute** - Full control, requires setup
- **DigitalOcean App Platform** - Simple container deployment

---

## Step 2: Set Up the Android Project

### 2.1 Create a new Android project

1. Open Android Studio
2. File > New > New Project
3. Select "Empty Compose Activity"
4. Configure:
   - Name: `StockMind`
   - Package: `com.stockmind.ai`
   - Language: Kotlin
   - Minimum SDK: API 26 (Android 8.0)
   - Build configuration: Kotlin DSL (Recommended)

### 2.2 Add dependencies in `build.gradle.kts` (app level)

```kotlin
dependencies {
    // Jetpack Compose (already included with template)
    implementation(platform("androidx.compose:compose-bom:2024.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.8.2")

    // Navigation
    implementation("androidx.navigation:navigation-compose:2.7.7")

    // Networking
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")

    // Charts
    implementation("com.patrykandpatrick.vico:compose-m3:1.13.1")

    // DataStore (local storage for auth tokens)
    implementation("androidx.datastore:datastore-preferences:1.0.0")

    // Dependency Injection
    implementation("io.insert-koin:koin-androidx-compose:3.5.3")
}
```

---

## Step 3: Define the API Client Layer

### 3.1 Create the API interface

```kotlin
// app/src/main/java/com/stockmind/ai/data/api/StockMindApi.kt

package com.stockmind.ai.data.api

import retrofit2.http.*

data class LoginRequest(val email: String, val password: String)
data class TokenResponse(val token: String, val tier: String)

data class PredictionRequest(
    val ticker: String,
    val timeframes: List<String> = listOf("1d", "1wk")
)

data class PredictionResponse(
    val ticker: String,
    val timeframe: String,
    val signal: String,
    val confidence: Float,
    val model_accuracy: Float,
    val current_price: Float,
    val target_price: Float?
)

interface StockMindApi {

    @POST("api/v1/auth/login")
    suspend fun login(@Body request: LoginRequest): TokenResponse

    @POST("api/v1/predict")
    suspend fun predict(
        @Header("Authorization") token: String,
        @Body request: PredictionRequest
    ): List<PredictionResponse>

    @GET("api/v1/watchlist/scan")
    suspend fun scanWatchlist(
        @Header("Authorization") token: String
    ): List<PredictionResponse>
}
```

### 3.2 Set up Retrofit

```kotlin
// app/src/main/java/com/stockmind/ai/data/api/ApiClient.kt

package com.stockmind.ai.data.api

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    private const val BASE_URL = "https://your-api-server.com/"

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        })
        .build()

    val api: StockMindApi = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(StockMindApi::class.java)
}
```

---

## Step 4: Build the Repository and ViewModel

### 4.1 Repository

```kotlin
// app/src/main/java/com/stockmind/ai/data/repository/StockRepository.kt

package com.stockmind.ai.data.repository

import com.stockmind.ai.data.api.*

class StockRepository(private val api: StockMindApi) {

    suspend fun login(email: String, password: String): Result<TokenResponse> {
        return try {
            val response = api.login(LoginRequest(email, password))
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getPredictions(
        token: String,
        ticker: String,
        timeframes: List<String>
    ): Result<List<PredictionResponse>> {
        return try {
            val response = api.predict(
                "Bearer $token",
                PredictionRequest(ticker, timeframes)
            )
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun scanWatchlist(token: String): Result<List<PredictionResponse>> {
        return try {
            val response = api.scanWatchlist("Bearer $token")
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

### 4.2 ViewModel

```kotlin
// app/src/main/java/com/stockmind/ai/ui/viewmodel/PredictionViewModel.kt

package com.stockmind.ai.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.stockmind.ai.data.api.PredictionResponse
import com.stockmind.ai.data.repository.StockRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class PredictionUiState(
    val isLoading: Boolean = false,
    val predictions: List<PredictionResponse> = emptyList(),
    val error: String? = null,
    val token: String? = null,
    val tier: String = "free"
)

class PredictionViewModel(private val repository: StockRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(PredictionUiState())
    val uiState: StateFlow<PredictionUiState> = _uiState

    fun login(email: String, password: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            repository.login(email, password)
                .onSuccess { response ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        token = response.token,
                        tier = response.tier
                    )
                }
                .onFailure { e ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = e.message ?: "Login failed"
                    )
                }
        }
    }

    fun fetchPredictions(ticker: String) {
        val token = _uiState.value.token ?: return
        val timeframes = if (_uiState.value.tier == "premium") {
            listOf("1h", "1d", "1wk", "1mo")
        } else {
            listOf("1d", "1wk")
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            repository.getPredictions(token, ticker, timeframes)
                .onSuccess { predictions ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        predictions = predictions
                    )
                }
                .onFailure { e ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = e.message ?: "Prediction failed"
                    )
                }
        }
    }
}
```

---

## Step 5: Build the UI Screens

### 5.1 Login Screen

```kotlin
// app/src/main/java/com/stockmind/ai/ui/screens/LoginScreen.kt

package com.stockmind.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.stockmind.ai.ui.viewmodel.PredictionViewModel

@Composable
fun LoginScreen(
    viewModel: PredictionViewModel,
    onLoginSuccess: () -> Unit
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    val uiState by viewModel.uiState.collectAsState()

    LaunchedEffect(uiState.token) {
        if (uiState.token != null) onLoginSuccess()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("StockMind AI", style = MaterialTheme.typography.headlineLarge)
        Spacer(modifier = Modifier.height(8.dp))
        Text("AI-Powered Stock Predictions", style = MaterialTheme.typography.bodyMedium)
        Spacer(modifier = Modifier.height(32.dp))

        OutlinedTextField(
            value = email,
            onValueChange = { email = it },
            label = { Text("Email") },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(16.dp))

        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Password") },
            visualTransformation = PasswordVisualTransformation(),
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = { viewModel.login(email, password) },
            modifier = Modifier.fillMaxWidth(),
            enabled = !uiState.isLoading
        ) {
            if (uiState.isLoading) {
                CircularProgressIndicator(modifier = Modifier.size(20.dp))
            } else {
                Text("Sign In")
            }
        }

        uiState.error?.let { error ->
            Spacer(modifier = Modifier.height(16.dp))
            Text(error, color = MaterialTheme.colorScheme.error)
        }
    }
}
```

### 5.2 Prediction Screen (Main Dashboard)

```kotlin
// app/src/main/java/com/stockmind/ai/ui/screens/PredictionScreen.kt

package com.stockmind.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.stockmind.ai.data.api.PredictionResponse
import com.stockmind.ai.ui.viewmodel.PredictionViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PredictionScreen(viewModel: PredictionViewModel) {
    var ticker by remember { mutableStateOf("") }
    val uiState by viewModel.uiState.collectAsState()

    Column(modifier = Modifier.fillMaxSize()) {
        // Top bar
        TopAppBar(
            title = { Text("StockMind AI") },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primary,
                titleContentColor = MaterialTheme.colorScheme.onPrimary
            )
        )

        Column(modifier = Modifier.padding(16.dp)) {
            // Search bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = ticker,
                    onValueChange = { ticker = it.uppercase() },
                    label = { Text("Enter Stock Ticker") },
                    placeholder = { Text("e.g. AAPL, MSFT, NVDA") },
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = { viewModel.fetchPredictions(ticker) },
                    enabled = ticker.isNotBlank() && !uiState.isLoading
                ) {
                    Text("Analyze")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Loading indicator
            if (uiState.isLoading) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Spacer(modifier = Modifier.height(8.dp))
                Text("Training ML models and generating predictions...")
            }

            // Error
            uiState.error?.let { error ->
                Text(error, color = MaterialTheme.colorScheme.error)
            }

            // Results
            if (uiState.predictions.isNotEmpty()) {
                val price = uiState.predictions.first().current_price
                Text(
                    "Current Price: $${String.format("%.2f", price)}",
                    style = MaterialTheme.typography.titleLarge
                )
                Spacer(modifier = Modifier.height(16.dp))

                LazyColumn {
                    items(uiState.predictions) { prediction ->
                        PredictionCard(prediction)
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }
            }
        }
    }
}

@Composable
fun PredictionCard(prediction: PredictionResponse) {
    val signalColor = when (prediction.signal) {
        "BUY" -> Color(0xFF4CAF50)
        "SELL" -> Color(0xFFF44336)
        else -> Color(0xFFFF9800)
    }

    val timeframeLabel = when (prediction.timeframe) {
        "1h" -> "Hourly"
        "1d" -> "Daily"
        "1wk" -> "Weekly"
        "1mo" -> "Monthly"
        else -> prediction.timeframe
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = signalColor.copy(alpha = 0.1f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(timeframeLabel, style = MaterialTheme.typography.titleMedium)
                Text(
                    prediction.signal,
                    style = MaterialTheme.typography.titleMedium,
                    color = signalColor
                )
            }
            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Confidence", style = MaterialTheme.typography.bodySmall)
                    Text("${(prediction.confidence * 100).toInt()}%",
                         style = MaterialTheme.typography.bodyLarge)
                }
                Column {
                    Text("Model Accuracy", style = MaterialTheme.typography.bodySmall)
                    Text("${(prediction.model_accuracy * 100).toInt()}%",
                         style = MaterialTheme.typography.bodyLarge)
                }
                prediction.target_price?.let { target ->
                    Column {
                        Text("Target", style = MaterialTheme.typography.bodySmall)
                        Text("$${String.format("%.2f", target)}",
                             style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { prediction.confidence },
                modifier = Modifier.fillMaxWidth(),
                color = signalColor,
            )
        }
    }
}
```

---

## Step 6: Navigation Setup

```kotlin
// app/src/main/java/com/stockmind/ai/ui/navigation/AppNavigation.kt

package com.stockmind.ai.ui.navigation

import androidx.compose.runtime.*
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.stockmind.ai.ui.screens.LoginScreen
import com.stockmind.ai.ui.screens.PredictionScreen
import com.stockmind.ai.ui.viewmodel.PredictionViewModel

@Composable
fun AppNavigation(viewModel: PredictionViewModel) {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "login") {
        composable("login") {
            LoginScreen(
                viewModel = viewModel,
                onLoginSuccess = {
                    navController.navigate("predictions") {
                        popUpTo("login") { inclusive = true }
                    }
                }
            )
        }
        composable("predictions") {
            PredictionScreen(viewModel = viewModel)
        }
    }
}
```

---

## Step 7: Main Activity

```kotlin
// app/src/main/java/com/stockmind/ai/MainActivity.kt

package com.stockmind.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.stockmind.ai.data.api.ApiClient
import com.stockmind.ai.data.repository.StockRepository
import com.stockmind.ai.ui.navigation.AppNavigation
import com.stockmind.ai.ui.viewmodel.PredictionViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val repository = StockRepository(ApiClient.api)
        val viewModel = PredictionViewModel(repository)

        setContent {
            MaterialTheme {
                Surface {
                    AppNavigation(viewModel = viewModel)
                }
            }
        }
    }
}
```

---

## Step 8: Android Manifest Permissions

Add to `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

---

## Step 9: Add Stock Charts (Optional)

Use the Vico charting library to display candlestick-style charts:

```kotlin
// Example: Line chart of price history
@Composable
fun StockChart(prices: List<Float>) {
    // Use Vico's CartesianChartHost for line/candlestick charts
    // See: https://patrykandpatrick.com/vico/
}
```

For full candlestick charts, alternatives include:
- **MPAndroidChart** (Java, well-established)
- **Vico** (Compose-native, modern)
- **Custom Canvas drawing** (full control)

---

## Step 10: Build, Test, and Release

### 10.1 Local Testing

```bash
# Start the backend API
cd /path/to/StockMind-AI
python api_server.py

# In Android Studio, update BASE_URL to your local IP
# e.g., "http://10.0.2.2:8000/" (for emulator)
# Run the app on an emulator or device
```

### 10.2 Build Release APK

1. Android Studio > Build > Generate Signed Bundle/APK
2. Select APK
3. Create or use a keystore
4. Select release build variant
5. The APK will be in `app/release/app-release.apk`

### 10.3 Publish to Google Play Store

1. Create a Google Play Developer account ($25 one-time fee)
2. Go to Google Play Console > Create Application
3. Fill in the store listing (title, description, screenshots)
4. Upload the signed APK or AAB (Android App Bundle)
5. Set pricing (free with in-app subscription, or paid)
6. Submit for review

---

## Project File Structure

```
StockMind/
├── app/
│   ├── src/main/
│   │   ├── java/com/stockmind/ai/
│   │   │   ├── MainActivity.kt
│   │   │   ├── data/
│   │   │   │   ├── api/
│   │   │   │   │   ├── StockMindApi.kt
│   │   │   │   │   └── ApiClient.kt
│   │   │   │   └── repository/
│   │   │   │       └── StockRepository.kt
│   │   │   └── ui/
│   │   │       ├── navigation/
│   │   │       │   └── AppNavigation.kt
│   │   │       ├── screens/
│   │   │       │   ├── LoginScreen.kt
│   │   │       │   └── PredictionScreen.kt
│   │   │       └── viewmodel/
│   │   │           └── PredictionViewModel.kt
│   │   ├── res/
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
└── settings.gradle.kts
```

---

## Key Considerations

### Performance
- The ML model training happens server-side, so API calls may take 10-30 seconds per prediction
- Implement proper loading states and timeout handling in the Android app
- Consider caching predictions locally with Room database

### Security
- Store auth tokens in Android EncryptedSharedPreferences, not plain SharedPreferences
- Use HTTPS for all API calls
- Do not hardcode API keys in the Android app

### Monetization
- Use Google Play Billing Library for in-app subscriptions (instead of Stripe)
- Map to the same free/premium tiers as the web app

### Push Notifications
- Use Firebase Cloud Messaging (FCM) to send alerts from the Bot.py scanner
- When Bot.py detects a BUY/SELL signal, send an FCM push instead of (or in addition to) Telegram

### Offline Support
- Cache the last fetched predictions in a local Room database
- Show cached data when offline with a "last updated" timestamp
