--[[

	--------------------------------------------------------------------

	Version 0.4.0

	Aqwam's Tensor Library (TensorL)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our terms and conditions at:
	
	https://github.com/AqwamCreates/TensorL/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = {}

local function checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray)

	if (#dimensionSizeArray ~= #targetDimensionSizeArray) then return false end

	for i, size in ipairs(dimensionSizeArray) do

		if (size ~= targetDimensionSizeArray[i]) then return false end

	end

	return true

end

local function checkIfDimensionIsOutOfBounds(dimension, minimumNumberOfDimensions, maximumNumberOfDimensions)

	if (dimension < minimumNumberOfDimensions) or  (dimension > maximumNumberOfDimensions) then error("The dimension is out of bounds.") end

end

local function checkIfDimensionSizeIndexIsOutOfBounds(dimensionSizeIndex, minimumDimensionSizeIndex, maximumDimensionSizeIndex)

	if (dimensionSizeIndex < minimumDimensionSizeIndex) or (dimensionSizeIndex > maximumDimensionSizeIndex) then error("The dimension size index is out of bounds.") end

end

local function removeFirstValueFromArray(array)

	local newArray = {}

	for i = 2, #array, 1 do table.insert(newArray, array[i]) end

	return newArray

end

local function removeLastValueFromArray(array)

	local newArray = table.clone(array)

	table.remove(newArray, #newArray)

	return newArray

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.
	
	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}
	
	if (numberOfDimensions >= 2) then
		
		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)
		
		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionUsingOneTensor(functionToApply, tensor[i], remainingDimensionSizeArray) end
		
	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.
		
		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(tensor[i]) end
		
	else -- Sometimes the original tensor can be a number, so we must do the operation directly.
		
		newTensor = functionToApply(tensor)
		
	end

	return newTensor

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.
	
	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)
		
		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionUsingTwoTensors(functionToApply, tensor1[i], tensor2[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.
	
		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(tensor1[i], tensor2[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.
		
		newTensor = functionToApply(tensor1, tensor2)

	end

	return newTensor

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(scalar, tensor[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		newTensor = functionToApply(scalar, tensor)

	end

	return newTensor

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor[i], scalar, remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(tensor[i], scalar) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		newTensor = functionToApply(tensor, scalar)

	end

	return newTensor

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

		local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray)

		else

			return functionToApply(tensor, dimensionSizeArray)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

		local isFirstValueATensor = (type(tensor) == "table")
		
		local isSecondValueATensor = (type(otherTensor) == "table")

		if (isFirstValueATensor) and (isSecondValueATensor) then

			tensor, otherTensor = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor, otherTensor)
			
			local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (not isFirstValueATensor) and (isSecondValueATensor) then
			
			local dimensionSizeArray = AqwamTensorLibrary:getSize(otherTensor)

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)
			
		elseif (isFirstValueATensor) and (not isSecondValueATensor) then
			
			local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)
			
		else
			
			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

local function get2DTensorTextSpacing(tensor, dimensionSizeArray, textSpacingArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray = get2DTensorTextSpacing(tensor[i], remainingDimensionSizeArray, textSpacingArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray[i] = math.max(textSpacingArray[i], string.len(tostring(tensor[i]))) end

	end

	return textSpacingArray

end

function AqwamTensorLibrary:get2DTensorTextSpacing(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local sizeAtFinalDimension = dimensionSizeArray[numberOfDimensions]

	local textSpacingArray = table.create(sizeAtFinalDimension, 0)

	return get2DTensorTextSpacing(tensor, dimensionSizeArray, textSpacingArray)

end

local function generateTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end
		
		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)
		
		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. " "

		end

		text = text .. " }"

	end
	
	return text
	
end

function AqwamTensorLibrary:generateTensorString(tensor)

	local textSpacingArray = AqwamTensorLibrary:get2DTensorTextSpacing(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return generateTensorString(tensor, dimensionSizeArray, textSpacingArray, 1)

end

local function generateTensorStringWithComma(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end
		
		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorStringWithComma(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " }"

	end

	return text
	
end

function AqwamTensorLibrary:generateTensorStringWithComma(tensor)

	local textSpacingArray = AqwamTensorLibrary:get2DTensorTextSpacing(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	return generateTensorString(tensor, dimensionSizeArray, textSpacingArray, 1)

end

local function generatePortableTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end
		
		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generatePortableTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

		if (dimensionDepth > 1) then text = text .. "," end

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " },"

	end
	
	return text
	
end

function AqwamTensorLibrary:generatePortableTensorString(tensor)
	
	local textSpacingArray = AqwamTensorLibrary:get2DTensorTextSpacing(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return generatePortableTensorString(tensor, dimensionSizeArray, textSpacingArray, 1)

end

function AqwamTensorLibrary:printTensor(tensor)

	print("\n\n" .. AqwamTensorLibrary:generateTensorString(tensor) .. "\n\n")

end

function AqwamTensorLibrary:printTensorWithComma(tensor)

	print("\n\n" .. AqwamTensorLibrary:generateTensorStringWithComma(tensor) .. "\n\n")

end

function AqwamTensorLibrary:printPortableTensor(tensor)

	print("\n\n" .. AqwamTensorLibrary:generatePortableTensorString(tensor) .. "\n\n")

end

function AqwamTensorLibrary:truncate(tensor, numberOfDimensionsToTruncate)

	numberOfDimensionsToTruncate = numberOfDimensionsToTruncate or math.huge

	if (numberOfDimensionsToTruncate ~= math.huge) and (numberOfDimensionsToTruncate ~= nil) then

		local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
		
		for dimension = 1, numberOfDimensionsToTruncate, 1 do
			
			local size = dimensionSizeArray[dimension]
			
			if (size ~= 1) then error("Unable to truncate. Dimension " .. dimension .. " has the size of " .. size .. ".") end
			
		end

	end

	local truncatedTensor = deepCopyTable(tensor)

	for dimension = 1, numberOfDimensionsToTruncate, 1 do

		if (type(truncatedTensor) ~= "table") then break end

		if (#truncatedTensor ~= 1) then break end

		truncatedTensor = truncatedTensor[1]

	end

	return truncatedTensor

end


local function containAFalseBooleanInTensor(booleanTensor, dimensionSizeArray)

	local numberOfValues = dimensionSizeArray[1]

	local containsAFalseBoolean = true

	if (#dimensionSizeArray > 1) then

		for i = 1, numberOfValues do containsAFalseBoolean = containAFalseBooleanInTensor(booleanTensor[i]) end

	else

		for i = 1, numberOfValues do 

			containsAFalseBoolean = (containsAFalseBoolean == booleanTensor[i])

			if (not containsAFalseBoolean) then return false end

		end

	end

	return containsAFalseBoolean

end

local function getTensorDimensionSizeArrayString(dimensionSizeArray)

	local numberOfDimensions = #dimensionSizeArray

	local tensorDimensionSizeArrayString = "("

	for s, size in ipairs(dimensionSizeArray) do

		tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. size

		if (s ~= numberOfDimensions) then

			tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. ", "

		end

	end

	tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. ")"

	return tensorDimensionSizeArrayString

end

local function onBroadcastError(dimensionSizeArray1, dimensionSizeArray2)

	local tensor1DimensionSizeArrayString = getTensorDimensionSizeArrayString(dimensionSizeArray1)

	local tensor2DimensionSizeArrayString = getTensorDimensionSizeArrayString(dimensionSizeArray2)

	local errorMessage = "Unable to broadcast. " .. "Tensor 1 size: " .. tensor1DimensionSizeArrayString .. " Tensor 2 size: " .. tensor2DimensionSizeArrayString

	error(errorMessage)

end

local function inefficientExpand(tensor, dimensionSizeArray, targetDimensionSizeArray)
	
	-- Does not do the same thing with efficient expand function. This one looks at lower dimensions from the parent dimension and makes copy of those. Then the function will have to go through all newly copied lower dimensions and do the same thing again, which is inefficient.

	if checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray) then return deepCopyTable(tensor) end -- Do not remove this code even if the code below is related or function similar to this code. You will spend so much time fixing it if you forget that you have removed it.

	local numberOfDimensions = #dimensionSizeArray

	local dimensionSize = dimensionSizeArray[1]

	local targetDimensionSize = targetDimensionSizeArray[1]

	local nextDimensionSize = dimensionSizeArray[2]

	local nextTargetDimensionSize = targetDimensionSizeArray[2]

	local hasSameNextDimensionSize = (nextDimensionSize == nextTargetDimensionSize)

	local canNextDimensionBeExpanded = (nextDimensionSize == 1)

	local newTensor = {}

	if (not canNextDimensionBeExpanded) and (not hasSameNextDimensionSize) then

		error("Unable to expand.")

	elseif (numberOfDimensions > 1) and (not hasSameNextDimensionSize) then

		for i = 1, targetDimensionSize, 1 do

			newTensor[i] = {} 

			for j = 1, nextTargetDimensionSize, 1 do newTensor[i][j] = deepCopyTable(tensor[i][1]) end

		end

	elseif (numberOfDimensions > 2) and (hasSameNextDimensionSize) then -- Do not remove this code even if the code above is related or function similar to this code. You will spend so much time fixing it if you forget that you have removed it.

		newTensor = deepCopyTable(tensor)

	elseif (numberOfDimensions == 1) and (dimensionSize == 1) then

		for i = 1, targetDimensionSize, 1 do table.insert(newTensor, tensor[1]) end

	end

	if (numberOfDimensions > 1) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingTargetDimensionSizeArray = removeFirstValueFromArray(targetDimensionSizeArray)

		for i = 1, targetDimensionSizeArray[1], 1 do newTensor[i] = inefficientExpand(newTensor[i], remainingDimensionSizeArray, remainingTargetDimensionSizeArray) end

	end
	
end

function AqwamTensorLibrary:inefficientExpand(tensor, targetDimensionSizeArray)  

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return inefficientExpand(tensor, dimensionSizeArray, targetDimensionSizeArray)

end

local function expand(tensor, dimensionSizeArray, targetDimensionSizeArray)
	
	-- Does not do the same thing with inefficient expand function. This one expand at the lowest dimension first and then the parent dimension will make copy of this.

	local newTensor

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions >= 2) then

		newTensor = {}
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingTargetDimensionSizeArray = removeFirstValueFromArray(targetDimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = expand(tensor[i], remainingDimensionSizeArray, remainingTargetDimensionSizeArray) end

	else

		newTensor = deepCopyTable(tensor)  -- If the "(numberOfDimensions > 1)" from the first "if" statement does not run, it will return the original tensor. So we need to deep copy it.

	end

	local updatedDimensionSizeArray = AqwamTensorLibrary:getSize(newTensor) -- Need to call this again because we may have modified the tensor below it, thus changing the dimension size array.

	local dimensionSize = updatedDimensionSizeArray[1]

	local targetDimensionSize = targetDimensionSizeArray[1]

	local hasSameDimensionSize = (dimensionSize == targetDimensionSize)

	local canDimensionBeExpanded = (dimensionSize == 1)

	if (numberOfDimensions >= 1) and (not hasSameDimensionSize) and (canDimensionBeExpanded) then 

		local subTensor = newTensor[1]

		for i = 1, targetDimensionSize, 1 do newTensor[i] = deepCopyTable(subTensor) end

	elseif (not hasSameDimensionSize) and (not canDimensionBeExpanded) then

		error("Unable to expand.")

	end

	return newTensor
	
end

function AqwamTensorLibrary:expand(tensor, targetDimensionSizeArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	if checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray) then return deepCopyTable(tensor) end -- Do not remove this code even if the code below is related or function similar to this code. You will spend so much time fixing it if you forget that you have removed it.

	return expand(tensor, dimensionSizeArray, targetDimensionSizeArray)

end

function AqwamTensorLibrary:increaseNumberOfDimensions(tensor, dimensionSizeToAddArray)

	local newTensor = {}

	local numberOfDimensionsToAdd = #dimensionSizeToAddArray

	if (numberOfDimensionsToAdd > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeToAddArray)

		for i = 1, dimensionSizeToAddArray[1], 1 do newTensor[i] = AqwamTensorLibrary:increaseNumberOfDimensions(tensor, remainingDimensionSizeArray) end

	elseif (numberOfDimensionsToAdd == 1) then

		for i = 1, dimensionSizeToAddArray[1], 1 do newTensor[i] = deepCopyTable(tensor) end

	else

		newTensor = tensor

	end

	return newTensor

end

local function getTheDimensionSizeArrayWithFewestNumberOfDimensionSizeOf1(dimensionSizeArray1, dimensionSizeArray2)

	local dimensionSizeOf1Count1 = 0

	local dimensionSizeOf1Count2 = 0

	for i = 1, #dimensionSizeArray1, 1 do

		if (dimensionSizeArray1[i] == 1) then dimensionSizeOf1Count1 = dimensionSizeOf1Count1 + 1 end

		if (dimensionSizeArray2[i] == 1) then dimensionSizeOf1Count2 = dimensionSizeOf1Count2 + 1 end

	end

	if (dimensionSizeOf1Count1 == 0) then

		return 2

	elseif (dimensionSizeOf1Count2 == 0) then

		return 1

	end

	if (dimensionSizeOf1Count1 < dimensionSizeOf1Count2) then

		return 1

	else

		return 2

	end

end

function AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor1, tensor2)

	local dimensionSizeArray1 = AqwamTensorLibrary:getSize(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #dimensionSizeArray1 

	local numberOfDimensions2 = #dimensionSizeArray2

	local haveSameNumberOfDimensions = (numberOfDimensions1 == numberOfDimensions2) -- Currently, if the number of dimensions have the same size, the tensor containing dimension with smaller axis will not expand. See case when tensor sizes are (5, 3, 6) and (5, 1, 6). So we need to be explicit in our dimensionSizeArrayWithHighestNumberOfDimensions variable.

	local isTensor1HaveLessNumberOfDimensions = (numberOfDimensions1 < numberOfDimensions2)

	local tensorNumberWithLowestNumberOfDimensions = (haveSameNumberOfDimensions and getTheDimensionSizeArrayWithFewestNumberOfDimensionSizeOf1(dimensionSizeArray1, dimensionSizeArray2)) or (isTensor1HaveLessNumberOfDimensions and 1) or 2

	local tensorWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and tensor1) or tensor2

	local dimensionSizeArrayWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and dimensionSizeArray1) or dimensionSizeArray2

	local dimensionSizeArrayWithHighestNumberOfDimensions = ((not isTensor1HaveLessNumberOfDimensions) and dimensionSizeArray1) or dimensionSizeArray2

	local lowestNumberOfDimensions = #dimensionSizeArrayWithLowestNumberOfDimensions

	local highestNumberOfDimensions = #dimensionSizeArrayWithHighestNumberOfDimensions

	local numberOfDimensionDifferences = highestNumberOfDimensions - lowestNumberOfDimensions

	local truncatedDimensionSizeArrayWithHighestNumberOfDimensions = table.clone(dimensionSizeArrayWithHighestNumberOfDimensions)

	for i = 1, numberOfDimensionDifferences, 1 do -- We need to remove the extra dimensions from tensor with highest number of dimensions. The values are removed starting from the first so that we can compare the endings.

		table.remove(truncatedDimensionSizeArrayWithHighestNumberOfDimensions, 1)

	end

	for i, dimensionSize in ipairs(dimensionSizeArrayWithLowestNumberOfDimensions) do -- Check if the endings are equal so that we can broadcast one of the tensor. If the dimension size are not equal and neither have dimension size of 1, then we can't broadcast the tensor with the lowest number of dimensions.

		if (dimensionSize ~=  truncatedDimensionSizeArrayWithHighestNumberOfDimensions[i]) and (dimensionSize ~= 1) then onBroadcastError(dimensionSizeArray1, dimensionSizeArray2) end

	end

	local dimensionSizeToAddArray = {}

	for i = 1, numberOfDimensionDifferences, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArrayWithHighestNumberOfDimensions[i]) end -- Get the dimension sizes of the left part of dimension size array.

	local expandedTensor = AqwamTensorLibrary:increaseNumberOfDimensions(tensorWithLowestNumberOfDimensions, dimensionSizeToAddArray)

	expandedTensor = AqwamTensorLibrary:expand(expandedTensor, dimensionSizeArrayWithHighestNumberOfDimensions)

	if (tensorNumberWithLowestNumberOfDimensions == 1) then

		return expandedTensor, tensor2

	else

		return tensor1, expandedTensor

	end

end

local function createTensor(dimensionSizeArray, initialValue) -- Don't put dimension size array truncation here. It is needed for several operations like dot product. 

	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createTensor(remainingDimensionSizeArray, initialValue) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = initialValue end

	end

	return tensor

end

function AqwamTensorLibrary:createTensor(dimensionSizeArray, initialValue)

	initialValue = initialValue or 0

	return createTensor(dimensionSizeArray, initialValue)

end

local function createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomNormalTensor(remainingDimensionSizeArray, mean, standardDeviation) end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			local randomNumber1 = math.random()

			local randomNumber2 = math.random()

			local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

			tensor[i] = (zScore * standardDeviation) + mean

		end

	end

	return tensor

end

function AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	return createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

end

local function createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)
	
	local numberOfDimensions = #dimensionSizeArray

	local tensor = {}
	
	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomUniformTensor(remainingDimensionSizeArray, minimumValue, maximumValue) end

	elseif (numberOfDimensions == 1) and (minimumValue) and (maximumValue) then
		
		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random(minimumValue, maximumValue) end
		
	elseif (numberOfDimensions == 1) and (minimumValue) and (not maximumValue) then
		
		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random(minimumValue) end
		
	elseif (numberOfDimensions == 1) and (not minimumValue) and (not maximumValue) then
		
		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random() end

	elseif (numberOfDimensions == 1) and (not minimumValue) and (maximumValue) then

		error("Invalid minimum value.")
		
	else
		
		error("An unknown error has occured when creating the random uniform tensor.")

	end

	return tensor

end

function AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	return createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

end

local function convertTensorToScalar(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (#dimensionSizeArray >= 1) then

		return convertTensorToScalar(tensor[1])

	else

		return tensor

	end

end

local function truncateDimensionSizeArrayIfRequired(dimensionSizeArray)

	local newDimensionSizeArray = table.clone(dimensionSizeArray)

	local numberOfStartingDimensionsWithTheSizeOf1 = 0

	while true do

		local size = newDimensionSizeArray[1]

		if (size ~= 1) then break end

		table.remove(newDimensionSizeArray, 1)

		numberOfStartingDimensionsWithTheSizeOf1 = numberOfStartingDimensionsWithTheSizeOf1 + 1

	end

	return newDimensionSizeArray, numberOfStartingDimensionsWithTheSizeOf1

end

local function createIdentityTensor(dimensionSizeArray, dimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	local tensor = {}

	if (numberOfDimensions >= 2) then

		for i = 1, dimensionSizeArray[1] do 

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			table.insert(copiedDimensionIndexArray, i)

			local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

			tensor[i] = createIdentityTensor(remainingDimensionSizeArray, copiedDimensionIndexArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			local firstDimensionIndex = copiedDimensionIndexArray[1]

			table.insert(copiedDimensionIndexArray, i)

			tensor[i] = 1

			for _, dimensionIndex in ipairs(copiedDimensionIndexArray) do

				if (dimensionIndex ~= firstDimensionIndex) then

					tensor[i] = 0
					break

				end

			end

		end

	end

	return tensor

end

function AqwamTensorLibrary:createIdentityTensor(dimensionSizeArray)

	local truncatedDimensionSizeArray, numberOfDimensionsOfSize1 = truncateDimensionSizeArrayIfRequired(dimensionSizeArray)
	
	local newTensor = createIdentityTensor(truncatedDimensionSizeArray, {})
	
	for i = 1, numberOfDimensionsOfSize1, 1 do newTensor = {newTensor} end

	return newTensor

end

local function getSize(tensor, sizeArray)
	
	if (type(tensor) ~= "table") then return end
		
	table.insert(sizeArray, #tensor)
		
	getSize(tensor[1], sizeArray)
	
end

function AqwamTensorLibrary:getSize(tensor)
	
	local dimensionSizeArray = {}
	
	getSize(tensor, dimensionSizeArray)

	return dimensionSizeArray

end

function AqwamTensorLibrary:getProperTensorFormatIfRequired(tensor)

	local resultTensorDimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (resultTensorDimensionSizeArray == nil) then return tensor end -- If our tensor is actually a scalar, just return the number.

	for _, size in ipairs(resultTensorDimensionSizeArray) do -- Return the original tensor if any dimension sizes are not equal to 1.

		if (size ~= 1) then return AqwamTensorLibrary:truncate(tensor) end

	end

	return convertTensorToScalar(tensor)

end

function AqwamTensorLibrary:getNumberOfDimensions(tensor)

	if (typeof(tensor) ~= "table") then return 0 end

	return AqwamTensorLibrary:getNumberOfDimensions(tensor[1]) + 1

end

local function getTotalSizeFromDimensionSizeArray(dimensionSizeArray)
	
	local totalSize = 1

	for _, value in ipairs(dimensionSizeArray) do totalSize = value * totalSize end
	
	return totalSize
	
end

function AqwamTensorLibrary:getTotalSize(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

end

--[[

local function transpose(originalTensor, tensor, targetTensor, dimensionIndexArray, currentDimension)

	currentDimension = currentDimension or 0

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	if (currentDimension ~= dimensionIndexArray[1]) then
		
		for i = 1, dimensionSizeArray[1], 1 do 
			
			table.insert(targetTensor, transpose(originalTensor, tensor[i], targetTensor, dimensionIndexArray, currentDimension + 1))
			
		end

	else
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = goToSubTensor(originalTensor, dimensionIndexArray[2]) end

		--for i = 1, dimensionSizeArray[1], 1 do   end

	end

	return targetTensor

end

--]]

--[[

local function transpose(originalTensor, dimensionIndexArray, currentDimension)
	
	currentDimension = currentDimension or 1
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(originalTensor)
	
	print(currentDimension == dimensionIndexArray[1])
	
	print(currentDimension)
	
	if (currentDimension == dimensionIndexArray[1]) then
		
		for i = 1, #dimensionSizeArray, 1 do originalTensor[i] = AqwamTensorLibrary:extract(originalTensor[i],  dimensionIndexArray) end
		
	else 
		
		for i = 1, #dimensionSizeArray, 1 do originalTensor[i] = transpose(originalTensor[i],  dimensionIndexArray, currentDimension + 1) end
		
	end
	
	return originalTensor
	
end

--]]

--[[

local function transpose(tensor, targetTensor, originDimensionIndex, targetDimensionIndex, currentDimension)
	
	currentDimension = currentDimension or 0
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	if (originDimensionIndex ~= currentDimension) then
		
		print(dimensionSizeArray[1])
		
		print(targetTensor)
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = goToSubTensor(tensor[i], targetDimensionIndex) end
		
	else
		
		for i = 1, dimensionSizeArray[1], 1 do targetTensor[i] = transpose(tensor[i], targetTensor[i], originDimensionIndex, targetDimensionIndex, currentDimension + 1) end
		
	end
	
	return targetTensor
	
end

--]]

local function hardcodedTranspose(tensor, targetDimensionArray) -- I don't think it is worth the effort to generalize to the rest of dimensions... That being said, to process videos, you need at most 5 dimensions. Don't get confused about the channels! Only number of channels are changed and not the number of dimensions of the tensor!

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionArray

	local offset = 5 - numberOfDimensions

	local dimensionSizeToAddArray = table.create(offset, 1)

	local expandedTensor = AqwamTensorLibrary:increaseNumberOfDimensions(tensor, dimensionSizeToAddArray)

	local targetDimension1 = targetDimensionArray[1] + offset
	local targetDimension2 = targetDimensionArray[2] + offset

	local expandedDimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor)
	
	targetDimensionArray = {targetDimension1, targetDimension2}

	expandedDimensionSizeArray[targetDimension1], expandedDimensionSizeArray[targetDimension2] = expandedDimensionSizeArray[targetDimension2], expandedDimensionSizeArray[targetDimension1]

	local newTensor = createTensor(expandedDimensionSizeArray, true)

	if (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 2)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[b][a][c][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[c][b][a][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 2)) and (table.find(targetDimensionArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][c][b][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[d][b][c][a][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[e][b][c][d][a]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 2)) and (table.find(targetDimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][d][c][b][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 2)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][e][c][d][b]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 3)) and (table.find(targetDimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][d][c][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 3)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][e][d][c]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 4)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][c][e][d]

						end

					end

				end

			end

		end

	else

		error("Invalid dimensions!")

	end

	return AqwamTensorLibrary:truncate(newTensor, offset)

end

function AqwamTensorLibrary:transpose(tensor, dimensionArray)

	if (AqwamTensorLibrary:getNumberOfDimensions(tensor) == 0) then return tensor end

	if (#dimensionArray ~= 2) then error("Dimension array must contain exactly 2 dimensions.") end

	if (dimensionArray[1] == dimensionArray[2]) then return tensor end

	return hardcodedTranspose(tensor, dimensionArray)

end

--[[

local function dotProduct(tensor1, tensor2) -- Second best one

	local tensor1DimensionSizeArray = AqwamTensorLibrary:getSize(tensor1)

	local tensor2DimensionSizeArray = AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #tensor1DimensionSizeArray

	local numberOfDimensions2 = #tensor2DimensionSizeArray

	local tensor = {}

	if (numberOfDimensions1 >= 3) and (numberOfDimensions2 >= 3) then

		for i = 1, tensor1DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2[i]) end
		
	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) then
		
		local tensor1Row = #tensor1
		
		local tensor1Column = #tensor1[1]
		
		local tensor2Column = #tensor2[1]
		
		for row = 1, tensor1Row, 1 do

			tensor[row] = {}

			for column = 1, tensor2Column, 1 do

				local sum = 0

				for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

				tensor[row][column] = sum

			end

		end

	elseif (numberOfDimensions1 == 1) and (numberOfDimensions2 >= 2) then

		for i = 1, tensor2DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1, tensor2[i]) end
		
	elseif (numberOfDimensions1 >= 2) and (numberOfDimensions2 == 1) then

		for i = 1, tensor2DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2) end

	elseif (numberOfDimensions1 == 1) and (numberOfDimensions2 == 1) then
		
		local sum = 0

		for i = 1, #tensor1 do sum = sum + tensor1[i] * tensor2[i] end

		tensor = sum 
		
	else
		
		print()
		
		error({numberOfDimensions1, numberOfDimensions2})

	end

	return tensor

end

--]]

--[[

local function dotProduct(tensor1, tensor2)

	local tensor1DimensionSizeArray = AqwamTensorLibrary:getSize(tensor1)

	local tensor2DimensionSizeArray = AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #tensor1DimensionSizeArray

	local numberOfDimensions2 = #tensor2DimensionSizeArray
	
	print(numberOfDimensions1, numberOfDimensions2)

	local tensor = {}
	
	if (numberOfDimensions1 == 1) and (numberOfDimensions2 == 1) then
		
		local sum = 0

		for i = 1, #tensor1 do sum = sum + tensor1[i] * tensor2[i] end

		tensor = sum 
		
	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) then

		local tensor1Row = #tensor1

		local tensor1Column = #tensor1[1]

		local tensor2Column = #tensor2[1]

		for row = 1, tensor1Row, 1 do

			tensor[row] = {}

			for column = 1, tensor2Column, 1 do

				local sum = 0

				for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

				tensor[row][column] = sum

			end

		end
		
	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 1) then
		
		for i = 1, tensor2DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2) end
		
	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 > 1) then
		
		for i = 1, tensor2DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2) end
		
	end

	return tensor

end

--]]

local function dotProduct(tensor1, tensor2, tensor1DimensionSizeArray, tensor2DimensionSizeArray) -- Best one. Do not delete!

	local numberOfDimensions1 = #tensor1DimensionSizeArray

	local numberOfDimensions2 = #tensor2DimensionSizeArray

	local tensor = {}

	if (numberOfDimensions1 == 1) and (numberOfDimensions2 == 2) then

		for i = 1, #tensor1, 1 do -- Last dimension, so represents columns.

			tensor[i] = 0

			for j = 1, #tensor2[1], 1 do tensor[i] = (tensor1[i] * tensor2[i][j]) end -- Since tensor 1 column size matches with tensor 2 row size, we can use column index from tensor 1.

		end

	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) then

		local tensor1Row = #tensor1

		local tensor1Column = #tensor1[1]

		local tensor2Column = #tensor2[1]

		for row = 1, tensor1Row, 1 do

			tensor[row] = {}

			for column = 1, tensor2Column, 1 do

				local sum = 0

				for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

				tensor[row][column] = sum

			end

		end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 > 2) then
		
		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)
		
		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2[i], remainingTensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 2) then
		
		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor = dotProduct(tensor1[i], tensor2, remainingTensor1DimensionSizeArray, tensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 == 1) and (numberOfDimensions2 > 2) then
		
		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor2DimensionSizeArray[1] do tensor = dotProduct(tensor1, tensor2[i], tensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 1) then

		for i = 1, tensor1DimensionSizeArray[1], 1 do

			for j = 1, tensor1DimensionSizeArray[2], 1 do 

				tensor[i] = {}

				local sum = 0

				for k = 1, tensor2DimensionSizeArray[1] do

					sum = sum + (tensor1[i][j] * tensor2[k]) 

				end

				tensor[i][j] = sum

			end

		end

	elseif (numberOfDimensions1 == 0) or (numberOfDimensions2 == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

	else

		error({numberOfDimensions1, numberOfDimensions2})

	end

	return tensor

end

local function tensor2DimensionalDotProduct(tensor1, tensor2)

	local subTensor = {}

	local tensor1Row = #tensor1

	local tensor1Column = #tensor1[1]

	local tensor2Row = #tensor2

	local tensor2Column = #tensor2[1]

	if (tensor1Column ~= tensor2Row) then error("Unable to perform the dot product. The size of second last dimension of the first tensor does not equal to the size of the last dimension of the second tensor.") end

	for row = 1, tensor1Row, 1 do

		subTensor[row] = {}

		for column = 1, tensor2Column, 1 do

			local sum = 0

			for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

			subTensor[row][column] = sum

		end

	end

	return subTensor

end

local function recursiveExpandedDotProduct(tensor1, tensor2, dimensionSizeArray1, dimensionSizeArray2) -- Since both have equal number of dimensions now, we only need to use only one dimension size array.

	local numberOfDimensions1 = #dimensionSizeArray1
	
	local numberOfDimensions2 = #dimensionSizeArray2

	local tensor

	if (numberOfDimensions1 >= 3) and (numberOfDimensions2 >= 3) and (dimensionSizeArray1[1] == dimensionSizeArray2[1]) then
		
		tensor = {}
		
		local remainingDimensionSizeArray1 = removeFirstValueFromArray(dimensionSizeArray1)
		
		local remainingDimensionSizeArray2 = removeFirstValueFromArray(dimensionSizeArray2)

		for i = 1, dimensionSizeArray1[1], 1 do tensor[i] = recursiveExpandedDotProduct(tensor1[i], tensor2[i], remainingDimensionSizeArray1, remainingDimensionSizeArray2) end

	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) and (dimensionSizeArray1[2] == dimensionSizeArray2[1]) then -- No need an elseif statement where number of dimension is 1. This operation requires 2D tensors.

		tensor = tensor2DimensionalDotProduct(tensor1, tensor2)

	elseif (numberOfDimensions1 == 0) or (numberOfDimensions2 == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)
		
	elseif (numberOfDimensions1 >= 2) and (numberOfDimensions2 >= 2) and (dimensionSizeArray1[1] ~= dimensionSizeArray2[1]) then
		
		error("Unable to dot product. The starting dimension sizes of the first tensor does not equal to the starting dimension sizes of the second tensor.")
		
	else
		
		error("Unable to dot product.")

	end

	return tensor

end

local function expandedDotProduct(tensor1, tensor2)
	
	local dimensionSizeArray1 =  AqwamTensorLibrary:getSize(tensor1)
	
	local dimensionSizeArray2 =  AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #dimensionSizeArray1

	local numberOfDimensions2 = #dimensionSizeArray2

	local highestNumberOfDimensions = math.max(numberOfDimensions1, numberOfDimensions2)

	local numberOfDimensionsOffset1 = highestNumberOfDimensions - numberOfDimensions1

	local numberOfDimensionsOffset2 = highestNumberOfDimensions - numberOfDimensions2

	local expandedTensor1

	local expandedTensor2
	
	if (numberOfDimensionsOffset1 ~= 0) then
		
		local dimensionSizeToAddArray = {}
		
		for i = 1, numberOfDimensionsOffset1, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray2[i]) end
		
		expandedTensor1 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor1, dimensionSizeToAddArray)
		
	else
		
		expandedTensor1 = tensor1
		
	end
	
	if (numberOfDimensionsOffset2 ~= 0) then
		
		local dimensionSizeToAddArray = {}
		
		for i = 1, numberOfDimensionsOffset2, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray1[i]) end

		expandedTensor2 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor2, dimensionSizeToAddArray)

	else
		
		expandedTensor2 = tensor2

	end
	
	local expandedTensor1DimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor1)
	
	local expandedTensor2DimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor2)

	return recursiveExpandedDotProduct(expandedTensor1, expandedTensor2, expandedTensor1DimensionSizeArray, expandedTensor2DimensionSizeArray)

end

local function hardcodedDotProduct(tensor1, tensor2)

	local numberOfDimensions1 = AqwamTensorLibrary:getNumberOfDimensions(tensor1)

	local numberOfDimensions2 = AqwamTensorLibrary:getNumberOfDimensions(tensor2)

	local numberOfDimensionsOffset1 = 5 - numberOfDimensions1

	local numberOfDimensionsOffset2 = 5 - numberOfDimensions2

	local expandedTensor1 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor1, table.create(numberOfDimensionsOffset1, 1))

	local expandedTensor2 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor2, table.create(numberOfDimensionsOffset2, 1))

	local expandedNumberOfDimension1 = AqwamTensorLibrary:getSize(expandedTensor1)

	local expandedNumberOfDimension2 = AqwamTensorLibrary:getSize(expandedTensor2)

	local tensor = {}

	for a = 1, expandedNumberOfDimension1[1], 1 do

		tensor[a] = {}

		for b = 1, expandedNumberOfDimension1[2], 1 do

			tensor[a][b] = {}

			for c = 1, expandedNumberOfDimension1[3], 1 do

				tensor[a][b][c] = {}

				for d = 1, expandedNumberOfDimension1[4], 1 do

					tensor[a][b][c][d] = {}

					for e = 1, expandedNumberOfDimension2[5], 1 do

						tensor[a][b][c][d][e] = {}

						local sum = 0

						for f = 1, expandedNumberOfDimension1[5] do sum = sum + (expandedTensor1[a][b][c][d][f] * expandedTensor2[a][b][c][f][e]) end

						tensor[a][b][c][d][e] = sum

					end

				end

			end

		end

	end

	return tensor

end

function AqwamTensorLibrary:dotProduct(...) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local tensorArray = {...}

	local tensor = tensorArray[1]

	for i = 2, #tensorArray, 1 do

		local otherTensor = tensorArray[i]

		tensor = expandedDotProduct(tensor, otherTensor)

	end

	return tensor

end

local function sumFromAllDimensions(tensor, dimensionSizeArray)

	local numberOfDimensions = #dimensionSizeArray

	local result = 0

	if (numberOfDimensions > 1) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do result = result + sumFromAllDimensions(tensor[i], remainingDimensionSizeArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do result = result + tensor[i] end

	end

	return result

end

local function recursiveSubTensorSumAlongFirstDimension(tensor, dimensionSizeArray, targetTensor, targetDimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions >= 1) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do

			local copiedTargetDimensionIndexArray = table.clone(targetDimensionIndexArray)

			table.insert(copiedTargetDimensionIndexArray, i)

			recursiveSubTensorSumAlongFirstDimension(tensor[i], remainingDimensionSizeArray, targetTensor, copiedTargetDimensionIndexArray)

		end

	else

		targetDimensionIndexArray[1] = 1 -- The target dimension only have a size of 1 for summing.

		local targetTensorValue = AqwamTensorLibrary:getValue(targetTensor, targetDimensionIndexArray)

		local value = targetTensorValue + tensor

		AqwamTensorLibrary:setValue(targetTensor, value, targetDimensionIndexArray)

	end	

end

local function subTensorSumAlongFirstDimension(tensor, dimensionSizeArray)

	local sumDimensionalSizeArray = table.clone(dimensionSizeArray)

	sumDimensionalSizeArray[1] = 1

	local sumTensor = createTensor(sumDimensionalSizeArray, 0)

	recursiveSubTensorSumAlongFirstDimension(tensor, dimensionSizeArray, sumTensor, {})

	return sumTensor

end

local function sumAlongOneDimension(tensor, dimensionSizeArray, targetDimension, currentDimension)

	local newTensor

	if (currentDimension == targetDimension) then

		newTensor = subTensorSumAlongFirstDimension(tensor, dimensionSizeArray)

	else

		newTensor = {}
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = sumAlongOneDimension(tensor[i], remainingDimensionSizeArray, targetDimension, currentDimension + 1) end

	end

	return newTensor

end

local function hardcodedDimensionSum(tensor, dimension) -- I don't think it is worth the effort to generalize to the rest of dimensions... That being said, to process videos, you need at most 5 dimensions. Don't get confused about the channels! Only number of channels are changed and not the number of dimensions of the tensor!

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionArray

	local offset = 5 - numberOfDimensions

	local dimension = dimension + offset

	local dimensionSizeToAddArray = table.create(offset, 1)

	local expandedTensor = AqwamTensorLibrary:expand(tensor, dimensionSizeToAddArray)

	local expandedDimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor)

	local expandedSumDimensionArray = table.clone(expandedDimensionSizeArray)

	expandedSumDimensionArray[dimension] = 1

	local newTensor = createTensor(expandedSumDimensionArray, 0)

	if (dimension == 1) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[1][b][c][d][e] = newTensor[1][b][c][d][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end

	elseif (dimension == 2) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][1][c][d][e] = newTensor[a][1][c][d][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end

	elseif (dimension == 3) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][1][d][e] = newTensor[a][b][1][d][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end

	elseif (dimension == 4) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][1][e] = newTensor[a][b][c][1][e] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end

	elseif (dimension == 5) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][1] = newTensor[a][b][c][d][1] + expandedTensor[a][b][c][d][e]

						end

					end

				end

			end

		end

	end

	return AqwamTensorLibrary:truncate(newTensor, offset)

end

function AqwamTensorLibrary:sum(tensor, dimension)
	
	dimension = dimension or 0
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local numberOfDimensions = #dimensionSizeArray

	if (dimension == 0) then return sumFromAllDimensions(tensor, dimensionSizeArray) end

	checkIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)
	
	local sumTensor = sumAlongOneDimension(tensor, dimensionSizeArray, dimension, 1)
	
	return sumTensor

end

function AqwamTensorLibrary:mean(tensor, dimension)

	local size = (dimension and AqwamTensorLibrary:getSize(tensor)[dimension]) or AqwamTensorLibrary:getTotalSize(tensor)

	local sumTensor = AqwamTensorLibrary:sum(tensor, dimension)

	local meanTensor = AqwamTensorLibrary:divide(sumTensor, size)

	return meanTensor

end

function AqwamTensorLibrary:standardDeviation(tensor, dimension)

	local size = (dimension and AqwamTensorLibrary:getSize(tensor)[dimension]) or AqwamTensorLibrary:getTotalSize(tensor)

	local meanTensor = AqwamTensorLibrary:mean(tensor, dimension)

	local subtractedTensor = AqwamTensorLibrary:subtract(tensor, meanTensor)

	local squaredSubractedTensor = AqwamTensorLibrary:power(subtractedTensor, 2)

	local summedSquaredSubtractedTensor = AqwamTensorLibrary:sum(squaredSubractedTensor, dimension)

	local squaredStandardDeviationTensor = AqwamTensorLibrary:divide(summedSquaredSubtractedTensor, size)

	local standardDeviationTensor = AqwamTensorLibrary:power(squaredSubractedTensor, 0.5)

	return standardDeviationTensor

end

function AqwamTensorLibrary:zScoreNormalization(tensor, dimension)

	local meanTensor = AqwamTensorLibrary:mean(tensor, dimension)

	local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(tensor, dimension)

	local subtractedTensor = AqwamTensorLibrary:subtract(tensor, meanTensor)

	local normalizedTensor = AqwamTensorLibrary:divide(subtractedTensor, standardDeviationTensor)

	return normalizedTensor, meanTensor, standardDeviationTensor

end

local function findMaximumValue(tensor, dimensionSizeArray)
	
	local highestValue = -math.huge

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			local value = AqwamTensorLibrary:findMaximumValue(tensor[i]) 

			highestValue = math.max(highestValue, value)

		end

	else

		highestValue = math.max(table.unpack(tensor))

	end
	
end

function AqwamTensorLibrary:findMaximumValue(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return findMaximumValue(tensor, dimensionSizeArray)

end

local function findMinimumValue(tensor, dimensionSizeArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local lowestValue = math.huge

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			local value = AqwamTensorLibrary:findMinimumValue(tensor[i]) 

			lowestValue = math.min(lowestValue, value)

		end

	else

		lowestValue = math.min(table.unpack(tensor))

	end
	
	return lowestValue
	
end

function AqwamTensorLibrary:findMinimumValue(tensor)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return findMinimumValue(tensor, dimensionSizeArray)

end

local function findMaximumValueDimensionIndexArray(tensor, dimensionSizeArray, dimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	local highestValue = -math.huge

	local highestValueDimensionIndexArray

	if (numberOfDimensions >= 2) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)
		
		for i = 1, dimensionSizeArray[1] do 

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			table.insert(copiedDimensionIndexArray, i)

			local subTensorHighestValueDimensionArray, value = findMaximumValueDimensionIndexArray(tensor[i], remainingDimensionSizeArray, dimensionIndexArray)

			if (value > highestValue) then

				highestValueDimensionIndexArray = table.clone(subTensorHighestValueDimensionArray)

				table.insert(highestValueDimensionIndexArray, i)

				highestValue = value

			end

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do

			local value = tensor[i]

			if (value > highestValue) then

				highestValueDimensionIndexArray = table.clone(dimensionIndexArray)

				table.insert(highestValueDimensionIndexArray, i)

				highestValue = value

			end

		end

	end

	return highestValueDimensionIndexArray, highestValue

end

function AqwamTensorLibrary:findMaximumValueDimensionIndexArray(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return findMaximumValueDimensionIndexArray(tensor, dimensionSizeArray, {})

end

local function findMinimumValueDimensionIndexArray(tensor, dimensionSizeArray, dimensionIndexArray)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local lowestValue = math.huge

	local lowestValueDimensionIndexArray

	if (numberOfDimensions >= 2) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1] do 

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			table.insert(copiedDimensionIndexArray, i)

			local subTensorLowestValueDimensionArray, value = findMinimumValueDimensionIndexArray(tensor[i], remainingDimensionSizeArray, dimensionIndexArray)

			if (value < lowestValue) then

				lowestValueDimensionIndexArray = table.clone(subTensorLowestValueDimensionArray)

				table.insert(lowestValueDimensionIndexArray, i)

				lowestValue = value

			end

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do

			local value = tensor[i]

			if (value < lowestValue) then

				lowestValueDimensionIndexArray = table.clone(dimensionIndexArray)

				table.insert(lowestValueDimensionIndexArray, i)

				lowestValue = value

			end

		end

	end

	return lowestValueDimensionIndexArray, lowestValue

end

function AqwamTensorLibrary:findMinimumValueDimensionIndexArray(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	return findMinimumValueDimensionIndexArray(tensor, dimensionSizeArray, {})

end

local function flattenIntoASingleDimension(tensor, dimensionSizeArray, targetTensor)

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do flattenIntoASingleDimension(tensor[i], remainingDimensionSizeArray, targetTensor) end

	else

		for _, value in ipairs(tensor) do table.insert(targetTensor, value) end

	end

	return tensor

end

local function flattenAlongSpecifiedDimensions(tensor, dimensionSizeArray, startDimension, endDimension)
	
	local numberOfDimensions = #dimensionSizeArray

	local flattenedDimensionSize = 1

	local newDimensionSizeArray = {}

	for currentDimension = numberOfDimensions, 1, -1 do

		local currentDimensionSize = dimensionSizeArray[currentDimension]

		if (currentDimension > startDimension) and (currentDimension <= endDimension) then 

			flattenedDimensionSize = flattenedDimensionSize * currentDimensionSize

		elseif (currentDimension == startDimension) then

			flattenedDimensionSize = flattenedDimensionSize * currentDimensionSize

			table.insert(newDimensionSizeArray, 1, flattenedDimensionSize)

		else

			table.insert(newDimensionSizeArray, 1, currentDimensionSize)

		end

	end
	
	return AqwamTensorLibrary:reshape(tensor, newDimensionSizeArray)
	
end

function AqwamTensorLibrary:flatten(tensor, startDimension, endDimension)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)
	
	local flattenedTensor
	
	if (not startDimension) and (not endDimension) then
		
		flattenedTensor = {}
		
		flattenIntoASingleDimension(tensor, dimensionSizeArray, flattenedTensor)
		
	else
		
		startDimension = startDimension or 1
		
		endDimension = endDimension or math.huge
		
		flattenedTensor = flattenAlongSpecifiedDimensions(tensor, dimensionSizeArray, startDimension, endDimension)
		
	end

	return flattenedTensor

end

local function reshapeFromFlattenedTensor(tensor, dimensionSizeArray, dimensionIndex)

	local newTensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			newTensor[i], dimensionIndex = reshapeFromFlattenedTensor(tensor, remainingDimensionSizeArray, dimensionIndex) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			table.insert(newTensor, tensor[dimensionIndex])
			dimensionIndex = dimensionIndex + 1

		end

	end

	return newTensor, dimensionIndex

end

local function incrementFinalDimensionIndex(targetDimensionSizeArray, currentTargetDimensionIndexArray)
	
	local numberOfDimensions = #currentTargetDimensionIndexArray
	
	currentTargetDimensionIndexArray[numberOfDimensions] = currentTargetDimensionIndexArray[numberOfDimensions] + 1
	
	for dimension = numberOfDimensions, 1, -1 do
		
		if ((targetDimensionSizeArray[dimension] + 1) == currentTargetDimensionIndexArray[dimension]) then
			
			currentTargetDimensionIndexArray[dimension] = 1
			
			if (dimension >= 2) then currentTargetDimensionIndexArray[dimension - 1] = currentTargetDimensionIndexArray[dimension - 1] + 1 end
			
		end	
		
	end
	
	return currentTargetDimensionIndexArray
	
end

local function reshape(tensor, dimensionSizeArray, targetTensor, targetDimensionSizeArray, currentTargetDimensionIndexArray)

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			currentTargetDimensionIndexArray = reshape(tensor[i], remainingDimensionSizeArray, targetTensor, targetDimensionSizeArray, currentTargetDimensionIndexArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 
			
			AqwamTensorLibrary:setValue(targetTensor, tensor[i], currentTargetDimensionIndexArray)
			
			currentTargetDimensionIndexArray = incrementFinalDimensionIndex(targetDimensionSizeArray, currentTargetDimensionIndexArray)

		end

	end
	
	return currentTargetDimensionIndexArray

end

function AqwamTensorLibrary:inefficientReshape(tensor, dimensionSizeArray) -- This one requires higher space complexity due to storing the target dimension index array for each of the values. It is also less efficient because it needs to use recursion to get and set values from and to the target tensor.
	
	local tensorDimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local totalNumberOfValue = getTotalSizeFromDimensionSizeArray(tensorDimensionSizeArray)

	local totalNumberOfValuesRequired = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	if (totalNumberOfValue ~= totalNumberOfValuesRequired) then error("The number of values of the tensor does not equal to total number of values of the reshaped tensor.") end

	local numberOfDimensions = #tensorDimensionSizeArray

	local newTensor

	if (numberOfDimensions == 1) then

		newTensor = reshapeFromFlattenedTensor(tensor, dimensionSizeArray, 1)

	else

		newTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, true)

		local currentTargetDimensionIndexArray = table.create(#dimensionSizeArray, 1)

		reshape(tensor, tensorDimensionSizeArray, newTensor, dimensionSizeArray, currentTargetDimensionIndexArray)

	end

	return newTensor
	
end

function AqwamTensorLibrary:reshape(tensor, dimensionSizeArray) -- This one requires lower space complexity as it only need to flatten the tensor. Then only need a single target dimension index array that will be used by all values from the original tebsor.

	local tensorDimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local totalNumberOfValue = getTotalSizeFromDimensionSizeArray(tensorDimensionSizeArray)

	local totalNumberOfValuesRequired = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	if (totalNumberOfValue ~= totalNumberOfValuesRequired) then error("The number of values of the tensor does not equal to total number of values of the reshaped tensor.") end

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions >= 2) then tensor = AqwamTensorLibrary:flatten(tensor) end
	
	local newTensor = reshapeFromFlattenedTensor(tensor, dimensionSizeArray, 1)

	return newTensor

end

local function getOutOfBoundsIndexArray(array, arrayToBeCheckedForOutOfBounds)

	local outOfBoundsIndexArray = {}

	for i, value in ipairs(arrayToBeCheckedForOutOfBounds) do

		if (value < 1) or (value > array[i]) then table.insert(outOfBoundsIndexArray, i) end

	end

	return outOfBoundsIndexArray

end

local function extract(tensor, dimensionSizeArray, originDimensionIndexArray, targetDimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	local extractedTensor = {}
	
	local originDimensionIndex = originDimensionIndexArray[1]
	
	local targetDimensionIndex = targetDimensionIndexArray[1]

	if (numberOfDimensions >= 2) and (originDimensionIndex <= targetDimensionIndex) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingOriginDimensionIndexArray = removeFirstValueFromArray(originDimensionIndexArray)

		local remainingTargetDimensionIndexArray = removeFirstValueFromArray(targetDimensionIndexArray)

		for i = originDimensionIndex, targetDimensionIndex, 1 do 

			local extractedSubTensor = extract(tensor[i], remainingDimensionSizeArray, remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray) 

			table.insert(extractedTensor, extractedSubTensor)

		end
		
	elseif (numberOfDimensions >= 2) and (originDimensionIndex > targetDimensionIndex) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingOriginDimensionIndexArray = removeFirstValueFromArray(originDimensionIndexArray)

		local remainingTargetDimensionIndexArray = removeFirstValueFromArray(targetDimensionIndexArray)
		
		for i = targetDimensionIndex, #tensor, 1 do 

			local extractedSubTensor = extract(tensor[i], remainingDimensionSizeArray, remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray) 

			table.insert(extractedTensor, extractedSubTensor)

		end

		for i = 1, originDimensionIndex, 1 do 

			local extractedSubTensor = extract(tensor[i], remainingDimensionSizeArray, remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray) 

			table.insert(extractedTensor, extractedSubTensor)

		end

	elseif (numberOfDimensions == 1) and (originDimensionIndex <= targetDimensionIndex) then

		for i = originDimensionIndex, targetDimensionIndex, 1 do table.insert(extractedTensor, tensor[i]) end
		
	elseif (numberOfDimensions == 1) and (originDimensionIndex > targetDimensionIndex) then
		
		for i = targetDimensionIndex, #tensor, 1 do table.insert(extractedTensor, tensor[i]) end

		for i = 1, originDimensionIndex, 1 do table.insert(extractedTensor, tensor[i]) end
		
	else
		
		error("An unknown error has occured while extracting the tensor.")

	end

	return extractedTensor

end

function AqwamTensorLibrary:extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #originDimensionIndexArray) then error("Invalid origin dimension index array.") end

	if (numberOfDimensions ~= #targetDimensionIndexArray) then error("Invalid target dimension index array.") end

	local outOfBoundsOriginIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, originDimensionIndexArray)

	local outOfBoundsTargetIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, targetDimensionIndexArray)

	local outOfBoundsOriginIndexArraySize = #outOfBoundsOriginIndexArray

	local outOfBoundsTargetIndexArraySize = #outOfBoundsTargetIndexArray

	if (outOfBoundsOriginIndexArraySize > 0) then

		local errorString = "Attempting to set an origin dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsOriginIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsOriginIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	if (outOfBoundsTargetIndexArraySize > 0) then

		local errorString = "Attempting to set an target dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsTargetIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsTargetIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	local extractedTensor = extract(tensor, dimensionSizeArray, originDimensionIndexArray, targetDimensionIndexArray)

	return extractedTensor

end

local function concatenate(targetTensor, otherTensor, targetDimensionSizeArray, targetDimension, currentDimension)

	if (currentDimension ~= targetDimension) then

		local remainingTargetDimensionSizeArray = removeFirstValueFromArray(targetDimensionSizeArray)

		for i = 1, targetDimensionSizeArray[1], 1 do targetTensor[i] = concatenate(targetTensor[i], otherTensor[i], remainingTargetDimensionSizeArray, targetDimension, currentDimension + 1) end

	else

		for _, value in ipairs(otherTensor) do table.insert(targetTensor, value) end

	end

	return targetTensor

end

function AqwamTensorLibrary:concatenate(tensor1, tensor2, dimension)
	
	if (type(dimension) ~= "number") then error("Invalid dimension.") end

	local dimensionSizeArray1 = AqwamTensorLibrary:getSize(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #dimensionSizeArray1

	local numberOfDimensions2 = #dimensionSizeArray2

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("The tensors do not have equal number of dimensions.") end

	if (numberOfDimensions1 <= 0) or (dimension > numberOfDimensions1) then error("The selected dimension is out of bounds.") end

	for dimensionIndex = 1, numberOfDimensions1, 1 do

		if (dimensionIndex == dimension) then continue end

		if (dimensionSizeArray1[dimensionIndex] ~= dimensionSizeArray2[dimensionIndex]) then error("The tensors do not contain equal dimension values at dimension " .. dimensionIndex .. ".") end

	end

	local targetTensor = deepCopyTable(tensor1)

	return concatenate(targetTensor, tensor2, dimensionSizeArray1, dimension, 1)

end

function AqwamTensorLibrary:add(...)

	return applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, ...)

end

function AqwamTensorLibrary:subtract(...)

	return applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, ...)

end

function AqwamTensorLibrary:multiply(...)

	return applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, ...)

end

function AqwamTensorLibrary:divide(...)

	return applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, ...)

end

function AqwamTensorLibrary:logarithm(...)

	return applyFunctionOnMultipleTensors(math.log, ...)

end

function AqwamTensorLibrary:exponent(...)

	return applyFunctionOnMultipleTensors(math.exp, ...)

end

function AqwamTensorLibrary:power(...)

	return applyFunctionOnMultipleTensors(math.pow, ...)

end

function AqwamTensorLibrary:isSameTensor(tensor1, tensor2)

	local booleanTensor = AqwamTensorLibrary:isEqualTo(tensor1, tensor2)

	return containAFalseBooleanInTensor(booleanTensor)

end

function AqwamTensorLibrary:isEqualTo(tensor1, tensor2)

	return applyFunctionUsingTwoTensors(function(a, b) return (a == b) end, tensor1, tensor2)

end

function AqwamTensorLibrary:isGreaterThan(tensor1, tensor2)

	return applyFunctionUsingTwoTensors(function(a, b) return (a > b) end, tensor1, tensor2)

end

function AqwamTensorLibrary:isGreaterOrEqualTo(tensor1, tensor2)

	return applyFunctionUsingTwoTensors(function(a, b) return (a >= b) end, tensor1, tensor2)

end

function AqwamTensorLibrary:isLessThan(tensor1, tensor2)

	return applyFunctionUsingTwoTensors(function(a, b) return (a < b) end, tensor1, tensor2)

end

function AqwamTensorLibrary:isLessOrEqualTo(tensor1, tensor2)

	return applyFunctionUsingTwoTensors(function(a, b) return (a <= b) end, tensor1, tensor2)

end

local function applyFunction(functionToApply, dimensionSizeArray, ...)

	local tensorArray = {...}

	local newTensor = {}

	if (#dimensionSizeArray >= 2) then
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			local subTensorArray = {}

			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, tensor[i]) end

			newTensor[i] = applyFunction(functionToApply, remainingDimensionSizeArray, table.unpack(subTensorArray)) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			local subTensorArray = {}

			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, tensor[i]) end

			newTensor[i] = functionToApply(table.unpack(subTensorArray)) 

		end

	end

	return newTensor

end

function AqwamTensorLibrary:applyFunction(functionToApply, ...)

	local tensorArray = {...}

	local allDimensionSizeArrays = {}

	for _, tensor in ipairs(tensorArray) do

		local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

		table.insert(allDimensionSizeArrays, dimensionSizeArray)

	end

	local firstDimensionSizeArray = allDimensionSizeArrays[1]

	for i = 2, #tensorArray, 1 do

		local dimensionSizeArray = allDimensionSizeArrays[i]

		if (#firstDimensionSizeArray ~= #dimensionSizeArray) then error("Tensor ".. (i - 1) .. " and " .. i .. " does not have the same number of dimensions.") end

		for s, size in ipairs(firstDimensionSizeArray) do

			if (size ~= dimensionSizeArray[s]) then error("Tensor " .. (i - 1) .. " and " .. i .. " does not contain equal dimension values at dimension " .. s .. ".") end

		end

	end
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensorArray[1])

	local resultTensor = applyFunction(functionToApply, dimensionSizeArray, ...)

	return resultTensor

end

local function setValue(tensor, dimensionSizeArray, value, dimensionIndexArray)
	
	local dimensionIndex = dimensionIndexArray[1]

	local numberOfDimensionIndices = #dimensionIndexArray

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensionIndices > numberOfDimensions) then

		error("The number of indices exceeds the tensor's number of dimensions.")

	elseif (numberOfDimensions >= 2) then

		checkIfDimensionSizeIndexIsOutOfBounds(dimensionIndex, 1, dimensionSizeArray[1])
		
		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingIndexArray = removeFirstValueFromArray(dimensionIndexArray)

		setValue(tensor[dimensionIndex], remainingDimensionSizeArray, value, remainingIndexArray)

	elseif (numberOfDimensions == 1) then

		tensor[dimensionIndex] = value

	else

		error("An error has occurred when attempting to set the tensor value.")

	end
	
end

function AqwamTensorLibrary:setValue(tensor, value, dimensionIndexArray)
	
	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	setValue(tensor, dimensionSizeArray, value, dimensionIndexArray)

end

function AqwamTensorLibrary:getValue(tensor, dimensionIndexArray)

	local dimensionIndex = dimensionIndexArray[1]

	local numberOfDimensionIndices = #dimensionIndexArray

	local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensionIndices > numberOfDimensions) then

		error("The number of indices exceeds the tensor's number of dimensions.")

	elseif (numberOfDimensions >= 2) then

		checkIfDimensionSizeIndexIsOutOfBounds(dimensionIndex, 1, dimensionSizeArray[1])

		local remainingIndexArray = removeFirstValueFromArray(dimensionIndexArray)

		return AqwamTensorLibrary:getValue(tensor[dimensionIndex], remainingIndexArray)

	elseif (numberOfDimensions == 1) then

		return tensor[dimensionIndex]

	else

		error("An error has occurred when attempting to get the tensor value.")

	end

end

function AqwamTensorLibrary:copy(tensor)

	return deepCopyTable(tensor)

end

return AqwamTensorLibrary
