# =====================================
# Part-2 : UAV Data Preprocess Script
# =====================================

require(lidR)
require(terra)
require(sf)
library(gstat)
library(raster)

las <- classify_noise(las, sor(15, 7))
las <- filter_poi(las, Classification != LASNOISE)

las <- classify_ground(las, algorithm = pmf(ws = 5, th = 3))
dtm_kriging <- rasterize_terrain(las, algorithm = kriging(k = 40))
las <- normalize_height(las, dtm_kriging)

chm <- grid_metrics(las, ~quantile(Z, probs = 0.98, na.rm = TRUE), res = 10)
chm <- terra::rast(chm)
ker <- matrix(1, 3, 3)
chm_smooth <- terra::focal(chm, w = ker, fun = median, na.rm = TRUE)

# Copyright (c) 2025 elainekalyn-10
# This work is licensed under the terms of the MIT license.  
# For a copy, see https://opensource.org/licenses/MIT
