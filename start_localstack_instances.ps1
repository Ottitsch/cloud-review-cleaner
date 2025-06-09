# LocalStack Multi-Instance Startup Script (PowerShell)
# This script starts 3 LocalStack instances for parallel processing

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start"
)

# Configuration for 3 LocalStack instances
$instances = @(
    @{
        Name = "instance-1"
        Port = 4566
        ContainerName = "localstack-instance-1"
        DataDir = ".localstack/instance-1"
        Services = "lambda,s3,dynamodb,ssm,iam,logs,events"
    },
    @{
        Name = "instance-2"
        Port = 4567
        ContainerName = "localstack-instance-2"
        DataDir = ".localstack/instance-2"
        Services = "lambda,s3,dynamodb,ssm,iam,logs,events"
    },
    @{
        Name = "instance-3"
        Port = 4568
        ContainerName = "localstack-instance-3"
        DataDir = ".localstack/instance-3"
        Services = "lambda,s3,dynamodb,ssm,iam,logs,events"
    }
)

function Test-DockerAvailable {
    try {
        $version = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Docker available: $version" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "[ERROR] Docker is not available or not running" -ForegroundColor Red
        Write-Host "   Please install Docker Desktop and ensure it's running" -ForegroundColor Yellow
        return $false
    }
    return $false
}

function New-DataDirectories {
    Write-Host "[INFO] Creating data directories..." -ForegroundColor Cyan
    foreach ($instance in $instances) {
        $dataDir = $instance.DataDir
        if (!(Test-Path $dataDir)) {
            New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
        }
        Write-Host "   [OK] Created: $dataDir" -ForegroundColor Green
    }
}

# Removed Stop-ExistingContainers function - containers now stay running

function Start-LocalStackInstance {
    param($instance)
    
    $containerName = $instance.ContainerName
    $port = $instance.Port
    
    # Create data directory if it doesn't exist
    if (!(Test-Path $instance.DataDir)) {
        New-Item -ItemType Directory -Path $instance.DataDir -Force | Out-Null
    }
    $dataDir = (Resolve-Path $instance.DataDir).Path
    $services = $instance.Services
    
    # Check if container is already running
    $existingContainer = docker ps --filter "name=$containerName" --format "{{.Names}}" 2>$null
    if ($existingContainer -eq $containerName) {
        Write-Host "[INFO] Container $containerName is already running on port $port" -ForegroundColor Yellow
        return $true
    }
    
    Write-Host "[START] Starting $($instance.Name) on port $port..." -ForegroundColor Cyan
    
    $dockerArgs = @(
        "run", "-d",
        "--name", $containerName,
        "-p", "$port`:4566",
        "-e", "SERVICES=$services",
        "-e", "DEBUG=1", 
        "-e", "LAMBDA_EXECUTOR=docker",
        "-e", "DOCKER_HOST=unix:///var/run/docker.sock",
        "-e", "DATA_DIR=/tmp/localstack-$($instance.Name)/data",
        "-e", "TMPDIR=/tmp/localstack-$($instance.Name)",
        "-e", "PORT_WEB_UI=$($port + 100)",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", "$dataDir`:/tmp/localstack-$($instance.Name)",
        "localstack/localstack:latest"
    )
    
    try {
        $containerId = docker @dockerArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   [OK] Started container: $containerId" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "   [ERROR] Failed to start $($instance.Name)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "   [ERROR] Failed to start $($instance.Name): $_" -ForegroundColor Red
        return $false
    }
}

function Wait-ForHealth {
    param($instance, $timeoutSeconds = 120)
    
    $port = $instance.Port
    $healthUrl = "http://localhost:$port/_localstack/health"
    
    Write-Host "[WAIT] Waiting for $($instance.Name) to become healthy..." -ForegroundColor Cyan
    
    $startTime = Get-Date
    while (((Get-Date) - $startTime).TotalSeconds -lt $timeoutSeconds) {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5 -ErrorAction Stop
            
            if ($response -and $response.services) {
                $requiredServices = $instance.Services -split ','
                $healthyServices = 0
                $unavailableServices = @()
                
                foreach ($service in $requiredServices) {
                    $serviceStatus = $null
                    # Handle PowerShell object property access properly
                    if ($response.services.PSObject.Properties.Name -contains $service) {
                        $serviceStatus = $response.services.$service
                    }
                    
                    # Consider both "running" and "available" as healthy
                    if ($serviceStatus -eq "running" -or $serviceStatus -eq "available") {
                        $healthyServices++
                    }
                    else {
                        $unavailableServices += "$service($serviceStatus)"
                    }
                }
                
                # Check if we have the minimum required services healthy
                $minimumRequired = [Math]::Max(1, [Math]::Floor($requiredServices.Count * 0.6)) # At least 60% of services
                
                if ($healthyServices -ge $minimumRequired) {
                    Write-Host "   [OK] $($instance.Name) is healthy!" -ForegroundColor Green
                    Write-Host "   [INFO] Healthy services: $healthyServices/$($requiredServices.Count)" -ForegroundColor Blue
                    if ($unavailableServices.Count -gt 0) {
                        Write-Host "   [WARN] Some services not ready: $($unavailableServices -join ', ')" -ForegroundColor Yellow
                    }
                    return $true
                }
                else {
                    Write-Host "   [WAIT] Still starting... $healthyServices/$($requiredServices.Count) services ready" -ForegroundColor Yellow
                    if ($unavailableServices.Count -gt 0) {
                        Write-Host "   [DEBUG] Waiting for: $($unavailableServices -join ', ')" -ForegroundColor DarkGray
                    }
                }
            }
        }
        catch {
            # Health endpoint not ready yet - this is normal during startup
            Write-Host "   [WAIT] Health endpoint not ready yet..." -ForegroundColor DarkGray
        }
        
        Start-Sleep -Seconds 3
    }
    
    Write-Host "   [ERROR] $($instance.Name) failed to become healthy within $timeoutSeconds seconds" -ForegroundColor Red
    return $false
}

function Start-AllInstances {
    if (!(Test-DockerAvailable)) {
        return $false
    }
    
    New-DataDirectories
    
    Write-Host "`n[START] Starting $($instances.Count) LocalStack instances..." -ForegroundColor Magenta
    
    $results = @()
    
    # Start all instances
    foreach ($instance in $instances) {
        $started = Start-LocalStackInstance -instance $instance
        $results += @{ Instance = $instance; Started = $started }
    }
    
    # Wait a bit for initialization
    Write-Host "[WAIT] Waiting for containers to initialize..." -ForegroundColor Cyan
    Start-Sleep -Seconds 3
    
    # Check health of each instance
    $healthyCount = 0
    foreach ($result in $results) {
        if ($result.Started) {
            $isHealthy = Wait-ForHealth -instance $result.Instance
            if ($isHealthy) {
                $healthyCount++
            }
        }
    }
    
    # Summary
    Write-Host "`n[SUCCESS] LocalStack Multi-Instance Setup Complete!" -ForegroundColor Magenta
    Write-Host "   [OK] Healthy instances: $healthyCount/$($instances.Count)" -ForegroundColor Green
    
    if ($healthyCount -gt 0) {
        Write-Host "`n[INFO] Instance URLs:" -ForegroundColor Cyan
        foreach ($instance in $instances) {
            $port = $instance.Port
            $name = $instance.Name
            Write-Host "   $name`: http://localhost:$port" -ForegroundColor Blue
            Write-Host "   Health: http://localhost:$port/_localstack/health" -ForegroundColor Blue
        }
        
        Write-Host "`n[NEXT] Ready for deployment! Run:" -ForegroundColor Green
        Write-Host "   python multi_deploy.py" -ForegroundColor Yellow
        
        Write-Host "`n[INFO] LocalStack instances are running in the background" -ForegroundColor Cyan
        Write-Host "   To stop them later, run: .\start_localstack_instances.ps1 -Action stop" -ForegroundColor Yellow
        return $true
    }
    else {
        Write-Host "`n[ERROR] No healthy instances available" -ForegroundColor Red
        return $false
    }
}

function Stop-AllInstances {
    Write-Host "[STOP] Stopping all LocalStack instances..." -ForegroundColor Cyan
    
    foreach ($instance in $instances) {
        $containerName = $instance.ContainerName
        try {
            docker stop $containerName 2>$null | Out-Null
            docker rm $containerName 2>$null | Out-Null
            Write-Host "   [OK] Stopped: $containerName" -ForegroundColor Green
        }
        catch {
            Write-Host "   [WARN] Error stopping $containerName" -ForegroundColor Yellow
        }
    }
    
    # Stop Docker Compose if exists
    if (Test-Path "docker-compose.localstack.yml") {
        try {
            docker-compose -f docker-compose.localstack.yml down 2>$null | Out-Null
            Write-Host "   [OK] Stopped Docker Compose services" -ForegroundColor Green
        }
        catch {
            Write-Host "   [WARN] Error stopping Docker Compose" -ForegroundColor Yellow
        }
    }
}

function Get-InstanceStatus {
    Write-Host "[STATUS] Checking LocalStack instance status...`n" -ForegroundColor Cyan
    
    foreach ($instance in $instances) {
        $name = $instance.Name
        $port = $instance.Port
        $containerName = $instance.ContainerName
        
        Write-Host "[CHECK] $name (port $port):" -ForegroundColor Blue
        
        # Check container status
        try {
            $containerInfo = docker ps --filter "name=$containerName" --format "table {{.Names}}\t{{.Status}}" 2>$null
            
            if ($containerInfo -match $containerName) {
                Write-Host "   [OK] Container: Running" -ForegroundColor Green
                
                # Check health endpoint
                try {
                    $healthUrl = "http://localhost:$port/_localstack/health"
                    $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5 -ErrorAction Stop
                    
                    if ($response -and $response.services) {
                        Write-Host "   [OK] Health: OK" -ForegroundColor Green
                        Write-Host "   [INFO] Services:" -ForegroundColor Blue
                        
                        foreach ($service in $response.services.PSObject.Properties) {
                            if ($service.Value -eq "running") {
                                Write-Host "      [OK] $($service.Name): $($service.Value)" -ForegroundColor Green
                            }
                            else {
                                Write-Host "      [ERROR] $($service.Name): $($service.Value)" -ForegroundColor Red
                            }
                        }
                    }
                    else {
                        Write-Host "   [ERROR] Health: Invalid response" -ForegroundColor Red
                    }
                }
                catch {
                    Write-Host "   [ERROR] Health: Not reachable ($_)" -ForegroundColor Red
                }
            }
            else {
                Write-Host "   [ERROR] Container: Not running" -ForegroundColor Red
                Write-Host "   [ERROR] Health: Not available" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "   [ERROR] Container: Status check failed" -ForegroundColor Red
        }
        
        Write-Host ""
    }
}

# Main execution
switch ($Action) {
    "start" {
        $success = Start-AllInstances
        if (!$success) {
            exit 1
        }
    }
    "stop" {
        Stop-AllInstances
    }
    "status" {
        Get-InstanceStatus
    }
}

Write-Host "[DONE] Script completed!" -ForegroundColor Green 