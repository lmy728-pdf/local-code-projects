# Offline restore of the versions used in this validation. Run from this directory.
dir.create('R_library',showWarnings=FALSE)
.libPaths(c(normalizePath('R_library'),.Library))
versions <- c(SparseM='1.84-2',MatrixModels='0.5-4',quadprog='1.5-8',quantreg='6.1',crs='0.15-46',np='0.80-1',jsonlite='2.0.0')
for(package in names(versions)) {
  path <- file.path('downloads','packages',paste0(package,'_',versions[[package]],'.zip'))
  stopifnot(file.exists(path))
  install.packages(path,repos=NULL,type='win.binary',lib=.libPaths()[1])
}
for(package in names(versions)) stopifnot(packageVersion(package)==package_version(versions[[package]]))
cat('Pinned packages restored.\n')
