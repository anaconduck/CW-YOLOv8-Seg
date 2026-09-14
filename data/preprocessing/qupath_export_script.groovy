/**
 * Export QuPath annotations to GeoJSON for YOLOv8 conversion.
 */

import qupath.lib.gui.QuPathGUI
import qupath.lib.io.GsonTools
import qupath.lib.objects.PathAnnotationObject

def currentImageData = getCurrentImageData()
def server = currentImageData.getServer()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

def project = QuPathGUI.getInstance().getProject()
def projectDir = project != null ? project.getPath().getParent().toString() : System.getProperty("user.home")
def exportFolder = new File(projectDir, "exported_annotations")

if (!exportFolder.exists()) {
    exportFolder.mkdirs()
}

def annotations = getAnnotationObjects()
def validClasses = ["necrosis", "normal", "steatosis"]
int matchedCount = 0

for (annotation in annotations) {
    def pathClass = annotation.getPathClass()
    if (pathClass != null) {
        def className = pathClass.getName().toLowerCase()
        for (target in validClasses) {
            if (className.contains(target)) {
                matchedCount++
                break
            }
        }
    }
}

println "Matched " + matchedCount + " annotations in " + imageName

def outputFile = new File(exportFolder, imageName + ".geojson")
def gson = GsonTools.getInstance(true)
def featureCollection = PathWriter.getFeatureCollection(annotations)

outputFile.withWriter('UTF-8') { writer ->
    gson.toJson(featureCollection, writer)
}

println "Exported: " + outputFile.getAbsolutePath()

