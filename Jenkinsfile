pipeline{
    agent{
        docker {
            image 'python:3.14-alpine'
        }
    }
    
    stages {
        stage('Install') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh 'python -m pytest'
            }
        }
        stage('Run') {
            steps {
                sh 'python main.py'
            }
        }
    }
}