@Library('podTemplateLib')

import net.santiment.utils.podTemplates

properties([
    buildDiscarder(
        logRotator(
            artifactDaysToKeepStr: '30',
            artifactNumToKeepStr: '',
            daysToKeepStr: '30',
            numToKeepStr: ''
        )
    )
])

slaveTemplates = new podTemplates()

slaveTemplates.dockerTemplate { label ->
    node(label) {
        stage('Build image') {
            container('docker') {
                def scmVars = checkout scm
                sh 'docker build -t llm-graph-builder-frontend:latest frontend'
                sh 'docker build -t llm-graph-builder-backend:latest backend'
            }
        }

        stage('Push image') {
            container('docker') {
                if (env.BRANCH_NAME == 'master' || env.BRANCH_NAME == 'main') {
                    withCredentials([
                        string(
                            credentialsId: 'aws_account_id',
                            variable: 'aws_account_id'
                        )
                    ]) {
                        def awsRegistry = "${env.aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com"
                        docker.withRegistry("https://${awsRegistry}", "ecr:eu-central-1:ecr-credentials") {
                            sh "docker tag llm-graph-builder-frontend:latest ${awsRegistry}/llm-graph-builder-frontend:latest"
                            sh "docker tag llm-graph-builder-backend:latest ${awsRegistry}/llm-graph-builder-backend:latest"
                            sh "docker push ${awsRegistry}/llm-graph-builder-frontend:latest"
                            sh "docker push ${awsRegistry}/llm-graph-builder-backend:latest"
                            sh "kubectl rollout restart deployment/llm-graph-builder"
                        }
                    }
                }
            }
        }
    }
}
