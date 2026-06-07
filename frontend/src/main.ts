import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <div class="container">
    <h1>House Price Prediction App</h1>
    <p class="subtitle">Enter house details and predict the estimated house price.</p>

    <form id="prediction-form" class="form-card">
      <div class="grid">
        <div>
          <label>Median Income</label>
          <input type="number" step="0.1" id="MedInc" value="8.3" required />
        </div>

        <div>
          <label>House Age</label>
          <input type="number" step="1" id="HouseAge" value="25" required />
        </div>

        <div>
          <label>Average Rooms</label>
          <input type="number" step="0.1" id="AveRooms" value="6.2" required />
        </div>

        <div>
          <label>Average Bedrooms</label>
          <input type="number" step="0.1" id="AveBedrms" value="1.1" required />
        </div>

        <div>
          <label>Population</label>
          <input type="number" step="1" id="Population" value="1200" required />
        </div>

        <div>
          <label>Average Occupancy</label>
          <input type="number" step="0.1" id="AveOccup" value="3.2" required />
        </div>

        <div>
          <label>Latitude</label>
          <input type="number" step="0.01" id="Latitude" value="34.05" required />
        </div>

        <div>
          <label>Longitude</label>
          <input type="number" step="0.01" id="Longitude" value="-118.25" required />
        </div>
      </div>

      <button type="submit">Predict Price</button>
    </form>

    <div id="result" class="result-card">
      Predicted price will appear here.
    </div>
  </div>
`

const form = document.querySelector<HTMLFormElement>('#prediction-form')!
const result = document.querySelector<HTMLDivElement>('#result')!

form.addEventListener('submit', async (event) => {
  event.preventDefault()

  result.innerHTML = 'Predicting...'

  const data = {
    MedInc: Number((document.querySelector<HTMLInputElement>('#MedInc')!).value),
    HouseAge: Number((document.querySelector<HTMLInputElement>('#HouseAge')!).value),
    AveRooms: Number((document.querySelector<HTMLInputElement>('#AveRooms')!).value),
    AveBedrms: Number((document.querySelector<HTMLInputElement>('#AveBedrms')!).value),
    Population: Number((document.querySelector<HTMLInputElement>('#Population')!).value),
    AveOccup: Number((document.querySelector<HTMLInputElement>('#AveOccup')!).value),
    Latitude: Number((document.querySelector<HTMLInputElement>('#Latitude')!).value),
    Longitude: Number((document.querySelector<HTMLInputElement>('#Longitude')!).value)
  }

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })

    const prediction = await response.json()

    result.innerHTML = `
      <h2>Predicted Price</h2>
      <p>$${prediction.predicted_price_usd.toLocaleString()}</p>
      <span>${prediction.message}</span>
    `
  } catch (error) {
    result.innerHTML = 'Error: Backend API is not running or connection failed.'
  }
})


