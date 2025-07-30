FROM ubuntu:22.04

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
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
EXPOSE 8087

# Run the service
CMD ["./build/mixture_models_service", "0.0.0.0", "8087"]