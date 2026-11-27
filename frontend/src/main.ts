import './style.css'

// ==================================================
// 1. APPLICATION CONFIGURATION
// ==================================================

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'
).replace(/\/+$/, '')

const REQUEST_TIMEOUT = 15000

// ==================================================
// 2. TYPESCRIPT INTERFACES
// ==================================================

interface HouseInput {
  MedInc: number
  HouseAge: number
  AveRooms: number
  AveBedrms: number
  Population: number
  AveOccup: number
  Latitude: number
  Longitude: number
}

interface PredictionResponse {
  predicted_price_usd: number
  message: string
}

interface FieldConfig {
  id: keyof HouseInput
  label: string
  description: string
  min: number
  max?: number
  step: string
  defaultValue: number
  exclusiveMin?: boolean
}

// ==================================================
// 3. FORM FIELD CONFIGURATION
// ==================================================

const fields: FieldConfig[] = [
  {
    id: 'MedInc',
    label: 'Median Household Income',
    description: 'Income in units of $10,000',
    min: 0,
    step: 'any',
    defaultValue: 8.3
  },
  {
    id: 'HouseAge',
    label: 'Median House Age',
    description: 'Age of houses in years',
    min: 0,
    step: 'any',
    defaultValue: 25
  },
  {
    id: 'AveRooms',
    label: 'Average Rooms',
    description: 'Average rooms per household',
    min: 0,
    exclusiveMin: true,
    step: 'any',
    defaultValue: 6.2
  },
  {
    id: 'AveBedrms',
    label: 'Average Bedrooms',
    description: 'Average bedrooms per household',
    min: 0,
    step: 'any',
    defaultValue: 1.1
  },
  {
    id: 'Population',
    label: 'Population',
    description: 'Population of the census block group',
    min: 0,
    exclusiveMin: true,
    step: '1',
    defaultValue: 1200
  },
  {
    id: 'AveOccup',
    label: 'Average Occupancy',
    description: 'Average household occupancy',
    min: 0,
    exclusiveMin: true,
    step: 'any',
    defaultValue: 3.2
  },
  {
    id: 'Latitude',
    label: 'Latitude',
    description: 'Geographical latitude',
    min: -90,
    max: 90,
    step: 'any',
    defaultValue: 34.05
  },
  {
    id: 'Longitude',
    label: 'Longitude',
    description: 'Geographical longitude',
    min: -180,
    max: 180,
    step: 'any',
    defaultValue: -118.25
  }
]

// ==================================================
// 4. APPLICATION INITIALISATION
// ==================================================

function getElement<T extends HTMLElement>(
  selector: string
): T {
  const element = document.querySelector<T>(selector)

  if (!element) {
    throw new Error(`Element not found: ${selector}`)
  }

  return element
}

const app = getElement<HTMLDivElement>('#app')

app.innerHTML = `
  <main class="container">

    <header class="app-header">

      <div class="brand-label">
        MACHINE LEARNING APPLICATION
      </div>

      <h1>House Price Prediction</h1>

      <p class="subtitle">
        Estimate California housing values using
        a Random Forest regression model.
      </p>

      <div class="tech-badges">
        <span>Python</span>
        <span>FastAPI</span>
        <span>TypeScript</span>
        <span>Random Forest</span>
      </div>

      <div
        id="api-status"
        class="api-status"
        role="status"
      >
        Checking backend connection...
      </div>

    </header>

    <section class="overview-grid">

      <article class="overview-card">
        <span>Prediction Model</span>
        <strong>Random Forest</strong>
      </article>

      <article class="overview-card">
        <span>Input Features</span>
        <strong>8</strong>
      </article>

      <article class="overview-card">
        <span>Dataset</span>
        <strong>California Housing</strong>
      </article>

      <article class="overview-card">
        <span>Output Currency</span>
        <strong>USD ($)</strong>
      </article>

    </section>

    <section class="workspace">

      <form
        id="prediction-form"
        class="form-card"
      >

        <div class="section-header">
          <h2>Property Information</h2>

          <p>
            Enter housing and location features
            to generate an estimated value.
          </p>
        </div>

        <div class="grid">

          ${fields.map(field => `
            <div class="form-group">

              <label for="${field.id}">
                ${field.label}
              </label>

              <input
                type="number"
                id="${field.id}"
                name="${field.id}"
                min="${field.min}"
                ${field.max !== undefined
                  ? `max="${field.max}"`
                  : ''}
                step="${field.step}"
                value="${field.defaultValue}"
                aria-describedby="${field.id}-hint"
                required
              />

              <small
                id="${field.id}-hint"
                class="field-hint"
              >
                ${field.description}
              </small>

            </div>
          `).join('')}

        </div>

        <div class="button-group">

          <button
            type="submit"
            id="predict-button"
            class="primary-button"
          >
            Predict House Price
          </button>

          <button
            type="reset"
            id="reset-button"
            class="secondary-button"
          >
            Reset
          </button>

        </div>

      </form>

      <section class="result-panel">

        <div
          id="result"
          class="result-card"
          role="status"
          aria-live="polite"
        >
          <h2>Prediction Result</h2>
          <p class="result-placeholder">
            Your estimated house price will appear here.
          </p>
        </div>

        <article class="information-card">

          <h3>How It Works</h3>

          <ol>
            <li>Enter the housing features.</li>
            <li>Validate the input data.</li>
            <li>Send the request to the FastAPI backend.</li>
            <li>Generate a Random Forest prediction.</li>
            <li>Display the estimated value in USD.</li>
          </ol>

        </article>

        <article class="information-card">

          <h3>Model Information</h3>

          <p>
            The prediction model is trained using
            the California Housing dataset.
          </p>

          <p class="disclaimer">
            Predictions are educational estimates based
            on historical census data, not current
            property valuations or financial advice.
          </p>

        </article>

      </section>

    </section>

    <footer class="app-footer">

      <p>
        House Price Prediction |
        Full-Stack Machine Learning Project
      </p>

      <p>
        Developed by Nikhil H R
      </p>

      <a
        href="https://github.com/NikhilH-R/house-price-prediction-devops"
        target="_blank"
        rel="noopener noreferrer"
      >
        View Source Code on GitHub
      </a>

    </footer>

  </main>
`

// ==================================================
// 5. DOM ELEMENTS
// ==================================================

const form = getElement<HTMLFormElement>(
  '#prediction-form'
)

const result = getElement<HTMLDivElement>(
  '#result'
)

const predictButton = getElement<HTMLButtonElement>(
  '#predict-button'
)

const resetButton = getElement<HTMLButtonElement>(
  '#reset-button'
)

const apiStatus = getElement<HTMLDivElement>(
  '#api-status'
)

let isPredicting = false

// ==================================================
// 6. API REQUEST WITH TIMEOUT
// ==================================================

async function requestAPI(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {

  const controller = new AbortController()

  const timeoutId = window.setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT
  )

  try {
    return await fetch(
      `${API_BASE_URL}${endpoint}`,
      {
        ...options,
        signal: controller.signal
      }
    )
  } finally {
    window.clearTimeout(timeoutId)
  }
}

// ==================================================
// 7. BACKEND HEALTH CHECK
// ==================================================

async function checkBackendHealth(): Promise<void> {

  apiStatus.textContent =
    'Checking backend connection...'

  apiStatus.className = 'api-status checking'

  try {
    const response = await requestAPI('/health')

    if (!response.ok) {
      throw new Error('Model unavailable')
    }

    const health = await response.json()

    if (
      health.status !== 'healthy' ||
      health.model_loaded !== true
    ) {
      throw new Error('Prediction service unavailable')
    }

    apiStatus.textContent =
      'Backend connected | Model ready'

    apiStatus.className = 'api-status online'

  } catch {
    apiStatus.textContent =
      'Backend unavailable | Check API connection'

    apiStatus.className = 'api-status offline'
  }
}

// ==================================================
// 8. INPUT VALIDATION
// ==================================================

function collectInput(): HouseInput {

  const values: Partial<HouseInput> = {}

  for (const field of fields) {

    const input = getElement<HTMLInputElement>(
      `#${field.id}`
    )

    const rawValue = input.value.trim()
    const value = Number(rawValue)

    if (
      rawValue === '' ||
      !Number.isFinite(value)
    ) {
      throw new Error(
        `${field.label} must be a valid number.`
      )
    }

    if (field.exclusiveMin && value <= field.min) {
      throw new Error(
        `${field.label} must be greater than ${field.min}.`
      )
    }

    if (value < field.min) {
      throw new Error(
        `${field.label} must be at least ${field.min}.`
      )
    }

    if (
      field.max !== undefined &&
      value > field.max
    ) {
      throw new Error(
        `${field.label} must not exceed ${field.max}.`
      )
    }

    values[field.id] = value
  }

  return values as HouseInput
}

// ==================================================
// 9. RESULT PRESENTATION
// ==================================================

function showMessage(
  title: string,
  message: string,
  state: 'loading' | 'error' | 'default'
): void {

  result.className = `result-card ${state}`

  const heading = document.createElement('h2')
  heading.textContent = title

  const paragraph = document.createElement('p')
  paragraph.textContent = message

  result.replaceChildren(heading, paragraph)
}

function showPrediction(
  prediction: PredictionResponse
): void {

  const formattedPrice = new Intl.NumberFormat(
    'en-US',
    {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 2
    }
  ).format(prediction.predicted_price_usd)

  result.className = 'result-card success'

  const heading = document.createElement('h2')
  heading.textContent = 'Estimated House Value'

  const price = document.createElement('p')
  price.className = 'prediction-price'
  price.textContent = formattedPrice

  const message = document.createElement('p')
  message.className = 'prediction-message'
  message.textContent = prediction.message

  const modelLabel = document.createElement('small')
  modelLabel.textContent =
    'Generated using Random Forest Regression'

  result.replaceChildren(
    heading,
    price,
    message,
    modelLabel
  )
}

// ==================================================
// 10. API RESPONSE VALIDATION
// ==================================================

function isPredictionResponse(
  data: unknown
): data is PredictionResponse {

  if (
    typeof data !== 'object' ||
    data === null
  ) {
    return false
  }

  const response = data as Record<string, unknown>

  return (
    typeof response.predicted_price_usd === 'number' &&
    Number.isFinite(response.predicted_price_usd) &&
    response.predicted_price_usd >= 0 &&
    typeof response.message === 'string'
  )
}

// ==================================================
// 11. LOADING STATE
// ==================================================

function setLoading(loading: boolean): void {

  isPredicting = loading

  predictButton.disabled = loading
  resetButton.disabled = loading
    form.querySelectorAll<HTMLInputElement>('input')
    .forEach(input => {
      input.disabled = loading
    })

  form.setAttribute(
    'aria-busy',
    String(loading)
  )

  predictButton.textContent = loading
    ? 'Generating Prediction...'
    : 'Predict House Price'
}

// ==================================================
// 12. HOUSE PRICE PREDICTION
// ==================================================

form.addEventListener(
  'submit',
  async (event) => {

    event.preventDefault()

    if (isPredicting) {
      return
    }

    try {

      const inputData = collectInput()

      setLoading(true)

      showMessage(
        'Generating Prediction',
        'Please wait while the model processes your input.',
        'loading'
      )

      const response = await requestAPI(
        '/predict',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(inputData)
        }
      )

      if (!response.ok) {

        if (response.status === 422) {
          throw new Error(
            'Invalid input. Please check the property details.'
          )
        }

        if (response.status === 503) {
          throw new Error(
            'The prediction model is currently unavailable.'
          )
        }

        throw new Error(
          `Prediction request failed (HTTP ${response.status}).`
        )
      }

      const prediction: unknown =
        await response.json()

      if (!isPredictionResponse(prediction)) {
        throw new Error(
          'The server returned an invalid prediction response.'
        )
      }

      showPrediction(prediction)

      apiStatus.textContent =
        'Backend connected | Model ready'

      apiStatus.className =
        'api-status online'

    } catch (error: unknown) {

      let message = 'Unable to generate a prediction.'

      if (
        error instanceof Error &&
        error.name === 'AbortError'
      ) {
        message =
          'The request timed out. Please try again.'

      } else if (error instanceof TypeError) {

        message =
          'Unable to connect to the backend API. ' +
          'Please ensure the FastAPI server is running.'

      } else if (error instanceof Error) {

        message = error.message
      }

      showMessage(
        'Prediction Unavailable',
        message,
        'error'
      )

      void checkBackendHealth()

      console.error(
        'Prediction request failed:',
        error
      )

    } finally {

      setLoading(false)
    }
  }
)

// ==================================================
// CLEAR STALE PREDICTION WHEN INPUT CHANGES
// ==================================================

form.addEventListener('input', () => {
  showMessage(
    'Prediction Result',
    'Input changed. Click Predict House Price to generate a new estimate.',
    'default'
  )
})

form.addEventListener('invalid', () => {
  showMessage(
    'Invalid Input',
    'Please enter valid property information before predicting.',
    'error'
  )
}, true)



// ==================================================
// 13. RESET FUNCTIONALITY
// ==================================================

form.addEventListener('reset', () => {

  showMessage(
    'Prediction Result',
    'Your estimated house price will appear here.',
    'default'
  )

  void checkBackendHealth()
})

// ==================================================
// 14. STARTUP HEALTH CHECK
// ==================================================

void checkBackendHealth()
