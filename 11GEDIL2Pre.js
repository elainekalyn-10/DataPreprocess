// =====================================
// Part-1 : GEDI L2A Data Extraction and Quality Control Script
// =====================================

var dataset_elevation = ee.Image('CGIAR/SRTM90_V4');
var elevation = dataset_elevation.select('elevation');
var slope = ee.Terrain.slope(elevation);

var qualityMask = function(im) {
  return im
    .updateMask(im.select('quality_flag').eq(1))          
    .updateMask(im.select('degrade_flag').eq(0))          
    .updateMask(im.select('sensitivity').gt(0.95))         
    .updateMask(slope.lte(10));                          
};

var gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
  .filterDate('2022-01-01', '2022-12-31')
  .filterBounds(geometry)
  .map(qualityMask)
  .select('rh98');

var clippedGedi = gedi.map(function(img) {
  return img.clip(geometry);
});
var dem = clippedGedi.mean().rename('rh98_canopy_height');

var points = dem.addBands(ee.Image.pixelLonLat()) 
  .sample({
    region: geometry,
    scale: 10,     
    geometries: true  
  });


Export.table.toDrive({
  collection: points,
  description: 'FuJian_CanopyHeight_2023_Enhanced_QC_Slope10',
  selectors: ['height', 'lat', 'lon'],
  fileFormat: 'CSV'
});

