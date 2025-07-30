FROM ubuntu:22.04

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Create build directory and build the service
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Expose port
EXPOSE 8088

# Run the service
CMD ["./build/special_functions_service", "0.0.0.0", "8088"]