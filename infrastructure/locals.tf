locals {
  tags = {
    Owner       = "gheegops"
    Project     = "gheegops"
    Environment = "${var.environment}"
    Toolkit     = "terraform"
    Name        = "${var.prefix}"
  }
}