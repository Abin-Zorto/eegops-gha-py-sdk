locals {
  tags = {
    Owner       = "Abin Zorto"
    Project     = "eegops"
    Environment = "${var.environment}"
    Toolkit     = "terraform"
    Name        = "${var.prefix}"
  }
}