import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export function AnalysisInsights() {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-lg font-medium">Key Insights</h3>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start">
            <span className="mr-2 text-primary">•</span>
            <span>
              Segment C shows the highest satisfaction (4.7/5) and loyalty, despite being only 15% of the market.
            </span>
          </li>
          <li className="flex items-start">
            <span className="mr-2 text-primary">•</span>
            <span>
              Segment B represents the largest portion of the market (40%) but has only medium loyalty and below-average
              spending.
            </span>
          </li>
          <li className="flex items-start">
            <span className="mr-2 text-primary">•</span>
            <span>
              There's a strong correlation between customer satisfaction and average spend across all segments.
            </span>
          </li>
        </ul>
      </div>

      <div>
        <h3 className="mb-2 text-lg font-medium">Business Recommendations</h3>
        <div className="space-y-3">
          <Card>
            <CardContent className="p-3">
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="bg-green-50 text-green-700">
                  High Priority
                </Badge>
                <div>
                  <p className="font-medium">Focus on Segment C expansion</p>
                  <p className="text-sm text-gray-600">
                    Develop targeted marketing campaigns to grow this high-value, loyal segment.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-3">
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="bg-blue-50 text-blue-700">
                  Medium Priority
                </Badge>
                <div>
                  <p className="font-medium">Improve Segment B retention</p>
                  <p className="text-sm text-gray-600">
                    Implement loyalty programs to increase satisfaction and average spend in this large segment.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-3">
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="bg-amber-50 text-amber-700">
                  Consider
                </Badge>
                <div>
                  <p className="font-medium">Evaluate Segment D strategy</p>
                  <p className="text-sm text-gray-600">
                    This segment shows low loyalty and spending. Consider repositioning or targeted offers.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
