
# Bias Detection Tool for Machine Learning Models

This web-based tool was developed as part of a thesis project at the University of Patras. Its goal is to detect and mitigate bias in machine learning models, providing users with an accessible platform to evaluate fairness metrics and employ bias mitigation algorithms. The tool aims to ensure that automated decision-making systems comply with fairness regulations, such as the NYC Bias Audit Law of 2021.

## Features

- **Bias Detection**: Analyze machine learning models for bias in key demographic attributes (e.g., race, gender, age).
- **Fairness Metrics**: Provides a range of fairness metrics to assess model performance.
- **Bias Mitigation**: Recommends and applies bias reduction algorithms based on identified issues.
- **Compliance**: Supports compliance with legal frameworks like the NYC Bias Audit Law.

## Technology Stack

- **Backend**: Python, Flask
- **Bias Detection Library**: IBM AIF360
- **Frontend**: HTML/CSS for a user-friendly interface

## How to Run

You can easily set up and run the tool using Docker and Docker Compose. Follow the steps below:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Build and start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

3. Access the tool via your web browser at `http://localhost:5000`.

## Future Extensions

- Support for additional fairness metrics
- Expanded bias mitigation techniques
- Broader compatibility with more machine learning models

---

© 2024 University of Patras - Developed as part of the Thesis of Orestis I. Tsagketas
