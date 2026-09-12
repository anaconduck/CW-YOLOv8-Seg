/**
 * QuPath Automated GeoJSON Annotation Exporter
 *
 * Designed for 2 Pathologists annotating Liver Histopathology:
 * Classes: "necrosis", "normal", "steatosis"
 *
 * How to use in QuPath:
 * 1. Open your project in QuPath.
 * 2. Go to: Automate -> Show script editor
 * 3. Paste this script into the editor.
 * 4. Go to: Run -> Run for project (or Run for current image).
 * 5. GeoJSON files will be exported directly into your project directory under "exported_annotations/".
 */

import qupath.lib.gui.QuPathGUI
import qupath.lib.io.GsonTools
import qupath.lib.objects.PathAnnotationObject

// 1. Define output directory
def currentImageData = getCurrentImageData()
def server = currentImageData.getServer()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

def project = QuPathGUI.getInstance().getProject()
def projectDir = project != null ? project.getPath().getParent().toString() : System.getProperty("user.home")
def exportFolder = new File(projectDir, "exported_annotations")

if (!exportFolder.exists()) {
    exportFolder.mkdirs()
}

// 2. Filter annotation objects
def annotations = getAnnotationObjects()
println "Found " + annotations.size() + " total annotations in " + imageName

// Verify classes
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

println "Matched " + matchedCount + " annotations for classes (necrosis, normal, steatosis)"

// 3. Export to GeoJSON
def outputFile = new File(exportFolder, imageName + ".geojson")

// Use QuPath GsonTools to serialize annotations as GeoJSON FeatureCollection
def gson = GsonTools.getInstance(true)
def featureCollection = PathWriter.getFeatureCollection(annotations)

outputFile.withWriter('UTF-8') { writer ->
    gson.toJson(featureCollection, writer)
}

println "[SUCCESS] Exported GeoJSON annotations to: " + outputFile.getAbsolutePath()
